import datetime
import os

import numpy as np
import torch
from tqdm import tqdm


def compute_metrics(preds, labels):
    """ Compute perplexity of a distribution """
    logp = torch.log(preds)
    logp = logp[labels != -100]
    nll = -logp.sum()
    return nll

def entropy(p):
    """ Compute the entropy of a probability distribution """
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)


# def print_1d_tensor(tensor):
#     if tensor.dtype != torch.long:
#         logger.info("\t" + "\t".join(f"{x:.5f}" for x in tensor.cpu().data))
#     else:
#         logger.info("\t" + "\t".join(f"{x:d}" for x in tensor.cpu().data))

# def print_2d_tensor(tensor):
#     """ Print a 2D tensor """
#     logger.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
#     for row in range(len(tensor)):
#         if tensor.dtype != torch.long:
#             logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
#         else:
#             logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))


def compute_heads_mlps_importance(
    args, model, eval_dataloader, compute_importance=True, head_mask = None, mlp_mask=None
):
    """ This method shows how to compute:
        - head attention entropy
        - head importance scores according to http://arxiv.org/abs/1905.10650
    """
    # Prepare our tensors
    n_layers, n_heads = model.bert.config.num_hidden_layers, model.bert.config.num_attention_heads
    mlp_importance = torch.zeros(n_layers).to(args.device)
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)

    if mlp_mask is None and compute_importance:
        mlp_mask = torch.ones(n_layers).to(args.device)
        mlp_mask.requires_grad_(requires_grad=True)
        head_mask = torch.ones(n_layers, n_heads).to(args.device)
        head_mask.requires_grad_(requires_grad=True)
    elif compute_importance:
        mlp_mask = mlp_mask.clone().detach()
        mlp_mask.requires_grad_(requires_grad=True)
        head_mask = head_mask.clone().detach()
        head_mask.requires_grad_(requires_grad=True)
    
    preds = None
    labels = None
    tot_tokens = 0.0
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(
            input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids, head_mask=head_mask, mlp_mask=mlp_mask
        )
        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  # Loss and logits are the first, attention the last
        loss.backward()  # Backpropagate to populate the gradients in the head mask
        if compute_importance:
            mlp_importance += mlp_mask.grad.abs().detach()
            head_importance += head_mask.grad.abs().detach()

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, label_ids.detach().cpu().numpy(), axis=0)

        tot_tokens += input_mask.float().detach().sum().data

    if compute_importance:
        # Normalize
        mlp_importance /= tot_tokens
        
        if not args.dont_normalize_importance_by_layer:
            exponent = 2
            norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
            head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

        if not args.dont_normalize_global_importance:
            mlp_importance = (mlp_importance - mlp_importance.min()) / (mlp_importance.max() - mlp_importance.min())
            head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())

        # Print/save matrices
        np.save(os.path.join(args.output_dir, "head_importance.npy"), head_importance.detach().cpu().numpy())
        np.save(os.path.join(args.output_dir, "mlp_importance.npy"), mlp_importance.detach().cpu().numpy())

        # logger.info("Layer importance scores")
        # print_1d_tensor(mlp_importance)
        # logger.info("MLP ranked by importance scores")
        # print_1d_tensor(mlp_importance.sort(descending=True)[1])


        # logger.info("Head importance scores")
        # print_2d_tensor(head_importance)
        # logger.info("Head ranked by importance scores")
        head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=args.device)
        head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
            head_importance.numel(), device=args.device
        )
        head_ranks = head_ranks.view_as(head_importance)
        # print_2d_tensor(head_ranks)
    return head_importance, mlp_importance, preds, labels


def mask_heads_mlps(args, model, eval_dataloader):
    """ This method shows how to mask head (set some heads to zero), to test the effect on the network,
        based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    head_importance, mlp_importance, preds, labels = compute_heads_mlps_importance(args, model, eval_dataloader)
    preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
    original_score = compute_metrics(args.task_name, preds, labels)[args.metric_name]
    # logger.info("Pruning: original score: %f, threshold: %f", original_score, original_score * args.masking_threshold)

    new_mlp_mask = torch.ones_like(mlp_importance)
    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = max(1, int(new_head_mask.numel() * args.masking_amount))
    
    # logger.info("Finding additional head masks")
    best_score = original_score
    current_score = best_score
    iteration = 0
    while current_score >= original_score * args.masking_threshold:
        best_score = current_score
        # Head New mask
        head_mask = new_head_mask.clone()  # save current head mask
        if args.save_mask_all_iterations:
            np.save(os.path.join(args.output_dir, f"head_mask_{iteration}.npy"), head_mask.detach().cpu().numpy())
            np.save(os.path.join(args.output_dir, f"head_importance_{iteration}.npy"), head_importance.detach().cpu().numpy())


        ###################### heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")

        current_heads_to_mask = head_importance.view(-1).sort()[1]

        # mask heads
        selected_heads_to_mask = []
        for head in current_heads_to_mask:
            if len(selected_heads_to_mask) == num_to_mask or head_importance.view(-1)[head.item()] == float("Inf"):
                break
            layer_idx = head.item() // model.bert.config.num_attention_heads
            head_idx = head.item() % model.bert.config.num_attention_heads
            new_head_mask[layer_idx][head_idx] = 0.0
            selected_heads_to_mask.append(head.item())
                
        if not selected_heads_to_mask:
            break

        # logger.info("Heads to mask: %s", str(selected_heads_to_mask))
        #new_head_mask = new_head_mask.view_as(head_mask)
        # print_2d_tensor(new_head_mask)
        
        ################################### MLP new mask
        mlp_mask = new_mlp_mask.clone()  # save current mlp mask
        if args.save_mask_all_iterations:
            np.save(os.path.join(args.output_dir, f"mlp_mask_{iteration}.npy"), mlp_mask.detach().cpu().numpy())
            np.save(os.path.join(args.output_dir, f"mlp_importance_{iteration}.npy"), mlp_importance.detach().cpu().numpy())

        iteration += 1

        # mlps from least important to most - keep only not-masked heads
        mlp_importance[mlp_mask == 0.0] = float("Inf")
        current_mlps_to_mask = mlp_importance.sort()[1]
        mlp_to_mask = current_mlps_to_mask[0]
        
        if mlp_importance[mlp_to_mask] == float("Inf"):
            break
        new_mlp_mask[mlp_to_mask] = 0.0
        # logger.info("MLP Layer to mask: %s", str(current_mlps_to_mask[0]))
        # print_1d_tensor(new_mlp_mask)

        # Compute metric and head,mlp importance again
        head_importance, mlp_importance, preds, labels = compute_heads_mlps_importance(
            args, model, eval_dataloader, head_mask=new_head_mask, mlp_mask=new_mlp_mask
        )

        preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
        current_score = compute_metrics(args.task_name, preds, labels)[args.metric_name]
        # logger.info(
        #     "MLP Masking: current score: %f, remaining mlps %d (%.1f percents)",
        #     current_score,
        #     new_mlp_mask.sum(),
        #     new_mlp_mask.sum() / new_mlp_mask.numel() * 100,
        # )
        # logger.info(
        #     "Head Masking: current score: %f, remaining heads %d (%.1f percents)",
        #     current_score,
        #     new_head_mask.sum(),
        #     new_head_mask.sum() / new_head_mask.numel() * 100,
        # )
    
    # logger.info("Finding additional head masks")
    current_score = best_score
    new_head_mask = head_mask
    # Only Heads
    while current_score >= original_score * args.masking_threshold:
        # Head New mask
        head_mask = new_head_mask.clone()  # save current head mask
        if args.save_mask_all_iterations:
            np.save(os.path.join(args.output_dir, f"head_mask_{iteration}.npy"), head_mask.detach().cpu().numpy())
            np.save(os.path.join(args.output_dir, f"head_importance_{iteration}.npy"), head_importance.detach().cpu().numpy())

        iteration += 1
        best_score = current_score
        ###################### heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        # mask heads
        selected_heads_to_mask = []
        for head in current_heads_to_mask:
            if len(selected_heads_to_mask) == num_to_mask // 2 or head_importance.view(-1)[head.item()] == float("Inf"):
                break
            layer_idx = head.item() // model.bert.config.num_attention_heads
            head_idx = head.item() % model.bert.config.num_attention_heads
            new_head_mask[layer_idx][head_idx] = 0.0
            selected_heads_to_mask.append(head.item())
                
        if not selected_heads_to_mask:
            break

        # logger.info("Heads to mask: %s", str(selected_heads_to_mask))
        # print_2d_tensor(new_head_mask)
        

        # Compute metric and head,mlp importance again
        head_importance, mlp_importance, preds, labels = compute_heads_mlps_importance(
            args, model, eval_dataloader, head_mask=new_head_mask, mlp_mask=mlp_mask
        )

        preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
        current_score = compute_metrics(args.task_name, preds, labels)[args.metric_name]
        # logger.info(
        #     "Head Masking: current score: %f, remaining heads %d (%.1f percents)",
        #     current_score,
        #     new_head_mask.sum(),
        #     new_head_mask.sum() / new_head_mask.numel() * 100,
        # )        
    
    # logger.info("Finding additional MLP masks")
    current_score = best_score
    new_mlp_mask = mlp_mask
    while current_score >= original_score * args.masking_threshold:            
        best_score = current_score
    
        ################################### MLP new mask
        mlp_mask = new_mlp_mask.clone()  # save current mlp mask
        if args.save_mask_all_iterations:
            np.save(os.path.join(args.output_dir, f"mlp_mask_{iteration}.npy"), mlp_mask.detach().cpu().numpy())
            np.save(os.path.join(args.output_dir, f"mlp_importance_{iteration}.npy"), mlp_importance.detach().cpu().numpy())

        iteration += 1
        # mlps from least important to most - keep only not-masked heads
        mlp_importance[mlp_mask == 0.0] = float("Inf")
        current_mlps_to_mask = mlp_importance.sort()[1]
        mlp_to_mask = current_mlps_to_mask[0]
        
        if mlp_importance[mlp_to_mask] == float("Inf"):
            break
        new_mlp_mask[mlp_to_mask] = 0.0
        # logger.info("MLP Layer to mask: %s", str(current_mlps_to_mask[0]))
        # print_1d_tensor(new_mlp_mask)

        # Compute metric and head,mlp importance again
        head_importance, mlp_importance, preds, labels = compute_heads_mlps_importance(
            args, model, eval_dataloader, head_mask=head_mask, mlp_mask=new_mlp_mask
        )

        preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
        current_score = compute_metrics(args.task_name, preds, labels)[args.metric_name]
        # logger.info(
        #     "MLP Masking: current score: %f, remaining mlps %d (%.1f percents)",
        #     current_score,
        #     new_mlp_mask.sum(),
        #     new_mlp_mask.sum() / new_mlp_mask.numel() * 100,
        # )

    # logger.info("Final head mask")
    # print_2d_tensor(head_mask)
    # logger.info("Final mlp mask")
    # print_1d_tensor(mlp_mask)
    np.save(os.path.join(args.output_dir, "head_mask.npy"), head_mask.detach().cpu().numpy())
    np.save(os.path.join(args.output_dir, "mlp_mask.npy"), mlp_mask.detach().cpu().numpy())

    return head_mask, mlp_mask


def prune_heads_mlps(args, model, eval_dataloader, head_mask, mlp_mask):
    """ This method shows how to prune head (remove heads weights) based on
        the head importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    # Try pruning and test time speedup
    # Pruning is like masking but we actually remove the masked weights
    before_time = datetime.now()
    _, _, preds, labels = compute_heads_mlps_importance(
        args, model, eval_dataloader, compute_importance=False, head_mask=head_mask, mlp_mask=mlp_mask
    )
    preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
    score_masking = compute_metrics(args.task_name, preds, labels)[args.metric_name]
    original_time = datetime.now() - before_time

    original_num_params = sum(p.numel() for p in model.parameters())
    
    heads_to_prune = {}
    for layer in range(len(head_mask)):
        heads_to_mask = [h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()]
        heads_to_prune[layer] = heads_to_mask
    assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
    # logger.info(f"{heads_to_prune}")
    model.prune_heads(heads_to_prune)
    
    
    mlps_to_prune = [h[0] for h in (1 - mlp_mask.long()).nonzero().tolist()]
    # logger.info(f"MLPS to prune - {mlps_to_prune}")
    model.prune_mlps(mlps_to_prune)

    
    pruned_num_params = sum(p.numel() for p in model.parameters())

    before_time = datetime.now()
    _, _, preds, labels = compute_heads_mlps_importance(
        args, model, eval_dataloader, compute_importance=False, head_mask=None, mlp_mask=None
    )
    preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
    score_pruning = compute_metrics(args.task_name, preds, labels)[args.metric_name]
    new_time = datetime.now() - before_time

    # logger.info(
    #     "Pruning: original num of params: %.2e, after pruning %.2e (%.1f percents)",
    #     original_num_params,
    #     pruned_num_params,
    #     pruned_num_params / original_num_params * 100,
    # )
    # logger.info("Pruning: score with masking: %f score with pruning: %f", score_masking, score_pruning)
    # logger.info("Pruning: speed ratio (original timing / new timing): %f percents", original_time / new_time * 100)
