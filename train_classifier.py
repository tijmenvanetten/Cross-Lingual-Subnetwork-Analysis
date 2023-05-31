import argparse
import json

import eval
import numpy as np
import torch
from transformers import (XLMRobertaForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments)

from data import *
from masking import prune_model
# from datasets.utils.logging import disable_progress_bar
# disable_progress_bar()
from models import ProbingClassificationHead
from language_props import num_labels_feature

# Setup evaluation 
import evaluate
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def train_classifier(args, model, dataset, data_collator, skip_train=False):
    training_args = TrainingArguments(
        output_dir=f"logs/results_classifier_{args.feature=}_{args.mask=}",
        overwrite_output_dir = 'True',
        evaluation_strategy="epoch",
        learning_rate=1e-3,
        num_train_epochs=10,
        push_to_hub=False,
        save_total_limit = 2,
        save_strategy = "epoch",
        remove_unused_columns=False,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_threshold=args.threshold, early_stopping_patience=args.patience)],
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print(f'Starting training.',)
    trainer.train()
    print(f'Evaluating trained classifier on test dataset',)
    test_results = trainer.evaluate(eval_dataset=dataset["test"])
    for lang in args.eval_langs:
        test_results[f'accuracy_{lang}'] = trainer.evaluate(eval_dataset=dataset[lang])

    with open('results.txt', 'a') as out:
        out.write(f"{args.feature=}, {args.mask=}")
        json.dump(test_results, out, indent=2)
        out.write('\n')

    if args.save_model:
        print("Saving model...")
        trainer.save_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_langs', nargs="*", default=['en', 'nl', 'he', 'hi', 'sw', 'cy'],
                        help='Language to finetune on.')
    parser.add_argument('--eval_langs', nargs="*", default=['fy', 'ar', 'ur', 'zu', 'gd'],
                        help='Language to finetune on.')
    parser.add_argument('--feature', default="writing_system", type=str,
                       help='Typological feature to classify: word_order, writing_system')                 
    parser.add_argument('--hidden_dim', default=100, type=int,
                       help='Hidden dimension size of the classifier')
    parser.add_argument('--pooling_type', default='cls', type=str,
                       help='Specify how sentence encodings are pooled')
    parser.add_argument('--model', default="xlm-roberta-base", type=str,
                       help='Pretrained model tokenizer to use.')
    parser.add_argument('--checkpoint', default="logs/results_['en', 'nl', 'fy', 'he', 'ar', 'hi', 'ur', 'sw', 'zu', 'cy', 'gd']", type=str,
                       help='Pretrained encoder to use.')
    parser.add_argument('--train_samples', default=10000, type=int,
                       help='Number of training samples per language')
    parser.add_argument('--eval_samples', default=1000, type=int,
                       help='Number of evaluation samples per language')
    parser.add_argument('--test_samples', default=9000, type=int,
                       help='Number of evaluation samples per language')
    parser.add_argument('--patience', default=3, type=int,
                       help='Number of epochs to wait before early stopping')
    parser.add_argument('--threshold', default=0.1, type=float,
                       help='Specify how much performance metric must improve before early stopping')
    parser.add_argument('--save_model', action='store_true',
                       help='Specify how much performance metric must improve before early stopping')
    parser.add_argument('--mask', default=None, type=str,
                        help='Mask to use for training')
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    print(f'Initializing model...', flush=True)
    # Load fine-tuned model into model for sequence classification
    num_labels = num_labels_feature[args.feature]
    model = XLMRobertaForSequenceClassification.from_pretrained(args.checkpoint, num_labels=num_labels)
    model.classifier = ProbingClassificationHead(input_size=768, hidden_size=args.hidden_dim, dropout=0.5, num_labels=num_labels, pooling=args.pooling_type)

    if args.mask:
        model = prune_model(args, model)

    # freeze encoder
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Prepare dataset
    print(f'Starting data preprocessing...', flush=True)
    cc100_typology_dataset = prepare_typology_dataset(args, tokenizer)
    cc100_typology_dataset.shuffle(12)
    print(cc100_typology_dataset['train'][:100])

    # Use end of sentence token as pad token
    tokenizer.pad_token = tokenizer.eos_token

    # Padding and batching
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # Training
    train_classifier(args, model, cc100_typology_dataset, data_collator)

