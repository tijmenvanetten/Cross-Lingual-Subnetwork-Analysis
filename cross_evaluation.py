import argparse
import json
import os

import numpy as np
import torch
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

from data import prepare_lm_dataset
from eval import evaluate_model
from masking import prune_model


def cross_evaluate_models(args, tokenizer, data_collator):
    """Evaluate the model for each language specific mask on each language."""
    print("Cross evaluating...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    training_args = TrainingArguments(
        output_dir=f'results_cross',
        overwrite_output_dir = 'False',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        num_train_epochs=10,
        weight_decay=0.01,
        push_to_hub=False,
        save_total_limit = 2,
        save_strategy = 'no',
        load_best_model_at_end=False
    )

    model = AutoModelForMaskedLM.from_pretrained(args.model)

    if args.mask_language is not 'None':
        args.mask = os.path.join(args.masks_dir, f"average_mask_{args.mask_language}.npy")
        model = prune_model(args, model)
    
    for dataset_language in args.eval_languages:
        print(f"Evaluating {args.mask_language} on {dataset_language}...")
        args.languages = dataset_language

        dataset = prepare_lm_dataset(args, tokenizer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['eval'],
            data_collator=data_collator
        )

        # eval_results = trainer.evaluate(eval_dataset=dataset['eval'])
        # print(eval_results)

        accuracy, perplexity = evaluate_model(args, model, trainer)
        print(f"Accuracy: {accuracy}, Perplexity: {perplexity}")

        with open(f"results_cross/{args.mask_language}_{dataset_language}.json", 'w') as f:
            json.dump({"accuracy": accuracy.item(), "perplexity": perplexity.item()}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks_dir", default="masks/head_masks", type=str,
                        help="Directory containing the masks")
    parser.add_argument("--model", default="xlm-roberta-base", type=str,
                        help="Model to use")
    parser.add_argument('--test_split', default=0.2, type=float,
                        help='Percentage of test split.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed.')
    parser.add_argument('--train_samples', default=100, type=int,
                        help='Number of training samples.')
    parser.add_argument('--eval_samples', default=5000, type=int,
                        help='Number of test samples.')
    parser.add_argument('--mask_language', default='en', type=str,
                        help='Language to mask.')
    parser.add_argument('--eval_languages', nargs="*", default=['en', 'nl', 'fy', 'he', 'ar', 'hi', 'ur', 'sw', 'zu', 'cy', 'gd'],
                        help='Languages to evaluate on.')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    cross_evaluate_models(args, tokenizer, data_collator)