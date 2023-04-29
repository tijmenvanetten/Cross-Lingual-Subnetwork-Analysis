"""
Dataset: https://huggingface.co/datasets/cc100
Based on: https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling
"""

# import lang2vec.lang2vec as l2v
from datasets import load_dataset
import argparse
from transformers import AutoTokenizer, XLMRobertaModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def load_dataset(args):
    return load_dataset("cc100", lang=args.languages, split="train[:10]")

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])

def group_texts(examples):
    block_size = 128
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--languages', default='nl', type=str,
                        help='Language to finetune on.')
    parser.add_argument('--debug', default=100, type=int,
                        help='Amount of samples to set up and debug the framework.')
    parser.add_argumen('--model', default="xlm-roberta-base", type=str,
                       help='Pretrained model to use.')
    
    args = parser.parse_args()
    cc100 = load_dataset(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = XLMRobertaModel.from_pretrained(args.model)

    tokenized_cc100 = cc100.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=cc100["train"].column_names,
        )
    
    lm_dataset = tokenized_cc100.map(group_texts, batched=True, num_proc=4)

    # Use end of sentence token as pad token
    tokenizer.pad_token = tokenizer.eos_token

    # Padding and batching
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=f"results_{args.languages}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(eval_results)