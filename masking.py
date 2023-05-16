import os
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm


def entropy(p):
    """ Compute the entropy of a probability distribution """
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)


def print_2d_tensor(tensor):
    """ Print a 2D tensor """
    print("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            print(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            print(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))


def compute_heads_importance(
    args, model, eval_dataloader, compute_entropy=True, compute_importance=True, head_mask=None
):
    """ This method shows how to compute:
        - head attention entropy
        - head importance scores according to http://arxiv.org/abs/1905.10650
    """
    # Prepare our tensors
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)
    attn_entropy = torch.zeros(n_layers, n_heads).to(args.device)

    if head_mask is None and compute_importance:
        head_mask = torch.ones(n_layers, n_heads).to(args.device)
        head_mask.requires_grad_(requires_grad=True)
    elif compute_importance:
        head_mask = head_mask.clone().detach()
        head_mask.requires_grad_(requires_grad=True)
    
    preds = None
    labels = None
    tot_tokens = 0.0
    accuracy_sum = 0.0
    # nlls = []
    total_count = 0

    print("***** Running evaluation *****")
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        batch = batch.to(args.device)
        # batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, label_ids = batch['input_ids'], batch['attention_mask'], batch['labels']

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(
            input_ids, attention_mask=input_mask, labels=label_ids, head_mask=head_mask
        )
        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  # Loss and logits are the first, attention the last
        # Print loss
        print(f"Loss at step {step}: {loss.item()}")

        loss.backward()  # Backpropagate to populate the gradients in the head mask

        if compute_entropy:
            for layer, attn in enumerate(all_attentions):
                masked_entropy = entropy(attn.detach()) * input_mask.float().unsqueeze(1)
                attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

        if compute_importance:
            head_importance += head_mask.grad.abs().detach()
            head_mask.grad.zero_()

        # nlls.append(loss)

        true_preds = (torch.argmax(logits, -1) == label_ids) * (label_ids != -100)
        # print(true_preds.sum() / (label_ids != -100).sum())
        accuracy_sum += true_preds.sum() / (label_ids != -100).sum()
        total_count += 1
            
        tot_tokens += input_mask.float().detach().sum().data

    # metric = torch.exp(torch.stack(nlls).mean())

    metric = accuracy_sum / total_count
    print(metric)

    return attn_entropy, head_importance, metric

# def compute_perplexity()

def mask_heads(args, model, eval_dataloader):
    """ This method shows how to mask head (set some heads to zero), to test the effect on the network,
        based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    _, head_importance, original_score = compute_heads_importance(args, model, eval_dataloader, compute_entropy=False)

    # original_score = compute_perplexity(preds, labels)
    print(f"Pruning: original score: {original_score}, threshold: {original_score * args.masking_threshold}")

    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = max(1, int(new_head_mask.numel() * args.masking_amount))

    # Copy original score
    current_score = original_score.copy_(original_score)
    i = 0
    # Change this when using perplexity
    while current_score >= original_score * args.masking_threshold:            
        head_mask = new_head_mask.clone()  # save current head mask
        if args.save_mask_all_iterations:
            np.save(os.path.join(args.output_dir, f"head_mask_{i}.npy"), head_mask.detach().cpu().numpy())
            np.save(os.path.join(args.output_dir, f"head_importance_{i}.npy"), head_importance.detach().cpu().numpy())
        i += 1
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        if len(current_heads_to_mask) <= num_to_mask:
            break

        # mask heads
        selected_heads_to_mask = []
        for head in current_heads_to_mask:
            if len(selected_heads_to_mask) == num_to_mask or head_importance.view(-1)[head.item()] == float("Inf"):
                break
            layer_idx = head.item() // model.config.num_attention_heads
            head_idx = head.item() % model.config.num_attention_heads
            new_head_mask[layer_idx][head_idx] = 0.0
            selected_heads_to_mask.append(head.item())
                
        if not selected_heads_to_mask:
            break

        print(f"Heads to mask: {str(selected_heads_to_mask)}")
        
        #new_head_mask = new_head_mask.view_as(head_mask)
        print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        _, head_importance, current_score = compute_heads_importance(
            args, model, eval_dataloader, compute_entropy=False, head_mask=new_head_mask
        )
        
        print(
            f"Masking: current score: {current_score}, remaning heads {new_head_mask.sum()} ({new_head_mask.sum() / new_head_mask.numel() * 100} percents)"
        )
        

    print("Final head mask")
    print_2d_tensor(new_head_mask)
    np.save(os.path.join(args.output_dir, "head_mask.npy"), new_head_mask.detach().cpu().numpy())

    return new_head_mask


def prune_heads(args, model, eval_dataloader, head_mask):
    """ This method shows how to prune head (remove heads weights) based on
        the head importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    # Try pruning and test time speedup
    # Pruning is like masking but we actually remove the masked weights
    before_time = datetime.now()
    _, head_importance, original_score = compute_heads_importance(
        args, model, eval_dataloader, compute_entropy=False, compute_importance=False, head_mask=head_mask
    )
    # preds = torch.argmax(preds, axis=-1) if args.output_mode == "classification" else np.squeeze(preds)
    score_masking = original_score
    original_time = datetime.now() - before_time

    original_num_params = sum(p.numel() for p in model.parameters())
    heads_to_prune = {}
    for layer in range(len(head_mask)):
        heads_to_mask = [h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()]
        heads_to_prune[layer] = heads_to_mask
    assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
    print(f"{heads_to_prune}")
    model.prune_heads(heads_to_prune)
    pruned_num_params = sum(p.numel() for p in model.parameters())

    before_time = datetime.now()
    _, head_importance, original_score = compute_heads_importance(
        args, model, eval_dataloader, compute_entropy=False, compute_importance=False, head_mask=None
    )
    # preds = torch.argmax(preds, axis=-1) if args.output_mode == "classification" else np.squeeze(preds)
    score_pruning = original_score
    new_time = datetime.now() - before_time

    print(
        f"Pruning: original num of params: {original_num_params:.2e}, after pruning {pruned_num_params:.2e} ({pruned_num_params / original_num_params * 100:.1f} percents)"
    )