import torch
from tqdm import tqdm

def evaluate(args, model, trainer, head_mask=None):
    dataloader = trainer.get_eval_dataloader()

    for batch in dataloader:
        print(batch['input_ids'][0])
        break

    losses = []
    accuracy_sum = 0
    total_count = 0

    for batch in tqdm(dataloader, desc="Iteration"):
        batch = batch.to(args.device)
        input_ids, input_mask, label_ids = batch['input_ids'], batch['attention_mask'], batch['labels']

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(
            input_ids, attention_mask=input_mask, labels=label_ids, head_mask=head_mask
        )
        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )
        losses.append(loss)
        true_preds = (torch.argmax(logits, -1) == label_ids) * (label_ids != -100)
        accuracy_sum += true_preds.sum() / (label_ids != -100).sum()
        total_count += 1

    accuracy = accuracy_sum / total_count
    perplexity = torch.exp(torch.stack(losses).mean())

    return accuracy, perplexity