import argparse
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                            DataCollatorForLanguageModeling, Trainer,
                            TrainingArguments)
from data import prepare_lm_dataset
import os
import numpy as np
import torch
import evaluate
from masking import prune_model
from eval import evaluate

metric = evaluate.load("accuracy", "f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


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
    print(f"Found masks: {mask_files}")
    languages = [lang[10:12] for lang in mask_files]
    # Create a dictionary of mask locations
    mask_dict = {lang: os.path.join(args.masks_dir, f"head_mask_{lang}.npy") for lang in languages}

    for mask_language in languages:
        model = AutoModelForMaskedLM.from_pretrained(args.model)
        args.mask = mask_dict[mask_language]

        model = prune_model(args, model)
        
        for dataset_language in languages:
            args.languages = dataset_language

            dataset = prepare_lm_dataset(args, tokenizer)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset['train'],
                eval_dataset=dataset['eval'],
                data_collator=data_collator,
                compute_metrics=compute_metrics
            )

            eval_results = trainer.evaluate(eval_dataset=dataset['eval'])
            print(eval_results)

            # accuracy, perplexity = evaluate(args, model, trainer, head_mask=head_mask)
            # print(f"Accuracy: {accuracy}, Perplexity: {perplexity}")

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

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    cross_evaluate_models(args, tokenizer, data_collator)