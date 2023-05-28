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
from eval import evaluate


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

    # Get the list of languages
    mask_files = os.listdir(args.masks_dir)
    mask_files = [mask_file for mask_file in mask_files if 'average' in mask_file]
    print(f"Found masks: {mask_files}")
    languages = [lang[13:15] for lang in mask_files]
    print(f"Languages: {languages}")
    # Create a dictionary of mask locations
    mask_dict = {lang: os.path.join(args.masks_dir, f"average_mask_{lang}.npy") for lang in languages}

    mask_languages = [None] + languages 

    for mask_language in mask_languages:
        model = AutoModelForMaskedLM.from_pretrained(args.model)

        if mask_language is not None:
            args.mask = mask_dict[mask_language]
            model = prune_model(args, model)
        
        for dataset_language in languages:
            print(f"Evaluating {mask_language} on {dataset_language}...")
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

            with open(f"results_cross/{mask_language}_{dataset_language}.json", 'w') as f:
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
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    cross_evaluate_models(args, tokenizer, data_collator)