import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, EarlyStoppingCallback
# from datasets.utils.logging import disable_progress_bar
# disable_progress_bar()

from data import *

def train(args, model, lm_dataset, data_collator):
    training_args = TrainingArguments(
        output_dir=f"results_{args.languages}",
        overwrite_output_dir = 'True',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=10,
        weight_decay=0.01,
        push_to_hub=False,
        save_total_limit = 2,
        save_strategy = "no",
        load_best_model_at_end=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=EarlyStoppingCallback(early_stopping_threshold=args.threshhold, early_stopping_patience=args.patience),
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["eval"],
        data_collator=data_collator
    )
    print(f'Started actual training.', flush=True)
    trainer.train()
    print(f'Finished training and start evaluating.', flush=True)
    eval_results = trainer.evaluate()
    print(eval_results, flush=True)
    print(f'Finished training and start evaluating.', flush=True)
    test_results = trainer.predict()
    print(eval_results, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--languages', default='nl', type=str,
                        help='Language to finetune on.')
    parser.add_argument('--model', default="xlm-roberta-base", type=str,
                       help='Pretrained model to use.')
    parser.add_argument('--num_samples', default=5000, type=int,
                       help='Number of data samples per language')
    parser.add_argument('--patience', default=3, type=int,
                       help='Number of epochs to wait before early stopping')
    parser.add_argument('--threshold', default=3, type=int,
                       help='Number of epochs to wait before early stopping')
    parser.add_argument('--train_split', default=0.8, type=float, choices=range(0,1),
                        help='Which percentage of the data to use for training')
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    # Prepare dataset
    print(f'Started data preprocessing...', flush=True)
    cc100_dataset = prepare_dataset(args, tokenizer)
    print(f'Finished data preprocessing.', flush=True)
    # Use end of sentence token as pad token
    tokenizer.pad_token = tokenizer.eos_token

    # Padding and batching
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # Training
    train(args, model, cc100_dataset, data_collator)

