import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, EarlyStoppingCallback
# from datasets.utils.logging import disable_progress_bar
# disable_progress_bar()

from data import *

def train(args, model, lm_dataset, data_collator):
    training_args = TrainingArguments(
        output_dir=f"logs/results_{args.languages}",
        overwrite_output_dir = 'True',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=10,
        weight_decay=0.01,
        push_to_hub=False,
        save_total_limit = 2,
        save_strategy = "epoch",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_threshold=args.threshold, early_stopping_patience=args.patience)],
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["eval"],
        data_collator=data_collator,
    )

    print(f'Evaluating pre-trained model...',)
    eval_results = trainer.evaluate()
    print(eval_results)

    print(f'Starting fine-tuning...',)
    trainer.train()

    print(f'Evaluating fine-tuned model...',)
    eval_results = trainer.evaluate()
    print(eval_results)

    print(f'Saving fine-tuned model...',)
    trainer.save_model()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--languages', nargs="*", default=['en', 'nl', 'fy', 'he', 'ar', 'hi', 'ur', 'sw', 'zu', 'cy', 'gd'],
                        help='Language to finetune on.')
    parser.add_argument('--model', default="xlm-roberta-base", type=str,
                       help='Pretrained model to use.')
    parser.add_argument('--train_samples', default=10000, type=int,
                       help='Number of training samples per language')
    parser.add_argument('--eval_samples', default=10000, type=int,
                       help='Number of evaluation samples per language')
    parser.add_argument('--patience', default=3, type=int,
                       help='Number of epochs to wait before early stopping')
    parser.add_argument('--threshold', default=0.1, type=float,
                       help='Specify how much performance metric must improve before early stopping')
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    # Prepare dataset
    print(f'Started data preprocessing...', flush=True)
    cc100_lm_dataset = prepare_lm_dataset(args, tokenizer)
    print(f'Finished data preprocessing.', flush=True)
    # Use end of sentence token as pad token
    tokenizer.pad_token = tokenizer.eos_token

    # Padding and batching
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # Training
    train(args, model, cc100_lm_dataset, data_collator)

