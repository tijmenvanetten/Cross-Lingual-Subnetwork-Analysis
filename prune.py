import argparse
import random

import numpy as np
import torch
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

from data import prepare_dataset
from masking import compute_heads_importance, mask_heads, prune_heads


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
        output_dir=f'logs/results_{args.languages}',
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
        eval_dataset=lm_dataset['eval'],
        data_collator=data_collator
    )

    print("Getting eval dataloader...")
    eval_dataloader = trainer.get_eval_dataloader()

    # Try head masking (set heads to zero until the score goes under a threshole)
    # and head pruning (remove masked heads and see the effect on the network)
    if args.try_masking and args.masking_threshold > 0.0 and args.masking_threshold < 1.0:
        head_mask = mask_heads(args, model, eval_dataloader)
        # Save head_mask as .npy file
        np.save(f'masks/head_mask_{args.languages}.npy', head_mask)
        # prune_heads(args, model, eval_dataloader, head_mask)
    else:
        #Compute head entropy and importance score
        compute_heads_importance(args, model, eval_dataloader)


    
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
    parser.add_argument('--try_masking', default=True, type=bool,
                        help='Whether to try masking heads and mlps.')
    parser.add_argument('--test_split', default=0.2, type=float,
                        help='Percentage of test split.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed.')
    parser.add_argument('--output_dir', default='results', type=str,
                        help='Output directory.')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Local rank.')
    parser.add_argument(
        "--dont_normalize_importance_by_layer", action="store_true", help="Don't normalize importance score by layers"
    )
    parser.add_argument(
        "--dont_normalize_global_importance",
        action="store_true",
        help="Don't normalize all importance scores between 0 and 1",
    )
    parser.add_argument(
        "--masking_amount", default=0.1, type=float, help="Amount to heads to masking at each masking step."
    )
    parser.add_argument('--data_dir', default='preprocessed_cc_100_fy', type=str,
                        help='Data directory.')
    parser.add_argument('--save_mask_all_iterations', default=True, type=bool,
                        help='Whether to save mask at each iteration.')
    parser.add_argument('--train_samples', default=4000, type=int,
                        help='Number of training samples.')
    parser.add_argument('--eval_samples', default=1000, type=int,
                        help='Number of test samples.')
    
    args = parser.parse_args()
    args.output_mode = 'classification'

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