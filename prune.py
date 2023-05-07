import argparse
import random

import numpy as np
import torch
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

from data import prepare_dataset
from masking import mask_heads_mlps, prune_heads_mlps


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def prune(args, model, lm_dataset, data_collator):
    print("Pruning...")
    set_seed(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    training_args = TrainingArguments(
        output_dir=f'results_{args.languages}',
        overwrite_output_dir = 'True',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        num_train_epochs=10,
        weight_decay=0.01,
        push_to_hub=False,
        save_total_limit = 2,
        save_strategy = 'no',
        load_best_model_at_end=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset['train'],
        eval_dataset=lm_dataset['test'],
        data_collator=data_collator
    )

    print("Getting eval dataloader...")
    eval_dataloader = trainer.get_eval_dataloader()

    # Print one example batch from dataloader
    for batch in eval_dataloader:
        print(batch)
        break

    print("Masking...")
    if args.try_masking and args.masking_threshold > 0.0 and args.masking_treshold < 1.0:
        head_mask, mlp_mask = mask_heads_mlps(args, model, eval_dataloader)
        prune_heads_mlps(args, model, eval_dataloader, head_mask, mlp_mask)
    # else:
    #     # Compute head entropy and importance score
    #     compute_heads_mlps_importance(args, model, eval_dataloader)


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--languages', default='nl', type=str,
                        help='Language to prune on.')
    parser.add_argument('--model', default="xlm-roberta-base", type=str,
                          help='Pretrained model to use.')
    parser.add_argument('--global_mask', default=None, type=str,
                        help='Global mask to use.')
    parser.add_argument('--masking_threshold', default=0.9, type=float, choices=range(0,1),
                        help='Threshold for masking heads and mlps.')
    parser.add_argument('--try_masking', default=False, type=bool,
                        help='Whether to try masking heads and mlps.')
    parser.add_argument('--test_split', default=0.2, type=float,
                        help='Percentage of test split.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed.')
    parser.add_argument('--output_dir', default='results', type=str,
                        help='Output directory.')
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    # Prepare dataset
    print("Preparing dataset...")
    cc100_dataset = prepare_dataset(args, tokenizer)

    # Use end of sentence token as pad token
    tokenizer.pad_token = tokenizer.eos_token

    # Padding and batching
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # Pruning
    prune(args, model, cc100_dataset, data_collator)