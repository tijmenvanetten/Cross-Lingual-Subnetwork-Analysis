import argparse
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                            DataCollatorForLanguageModeling, Trainer,
                            TrainingArguments)
from data import prepare_dataset
import os
import numpy as np
import torch
from evaluate import evaluate

def evaluate_model(model, mask, dataset, trainer):
    """Evaluate the model on the dataset."""
    print("Evaluating...")
    model.set_head_mask(mask)

    # Evaluate model for both perplexity and accuracy
    eval_results = trainer.evaluate(eval_dataset=dataset)
    print(eval_results)


def cross_evaluate_models(args, tokenizer, model, data_collator):
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
    print(f"Found masks: {mask_files}")
    languages = [lang[10:12] for lang in mask_files]
    # Create a dictionary of mask locations
    mask_dict = {lang: os.path.join(args.masks_dir, f"head_mask_{lang}.npy") for lang in languages}

    for mask_language in languages:
        mask_location = mask_dict[mask_language]
        head_mask = torch.from_numpy(np.load(mask_location))

        for dataset_language in languages:
            args.languages = dataset_language

            dataset = prepare_dataset(args, tokenizer)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset['train'],
                eval_dataset=dataset['eval'],
                data_collator=data_collator,
            )

            accuracy, perplexity = evaluate(args, model, trainer, head_mask=head_mask)
            print(f"Accuracy: {accuracy}, Perplexity: {perplexity}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks_dir", default="masks", type=str,
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
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    cross_evaluate_models(args, tokenizer, model, data_collator)