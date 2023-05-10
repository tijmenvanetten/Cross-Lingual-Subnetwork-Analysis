"""
Dataset: https://huggingface.co/datasets/cc100
Based on: https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling
"""
import argparse

from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer


def preprocess_function(examples, tokenizer):
    return tokenizer(examples['text'])

def group_texts(examples):
    block_size = 128

    # concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    concatenated_examples = sum(examples, [])

    # total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = len(concatenated_examples)
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    # Split by chunks of block_size.
    # result = {
    #     k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
    #     for k, t in concatenated_examples.items()
    # }
    result = [concatenated_examples[i : i + block_size] for i in range(0, total_length, block_size)]


    result["labels"] = result["input_ids"].copy()
    return result


def prepare_dataset(args, tokenizer):
    cc100 = load_dataset("cc100", lang=args.languages)


    tokenized_cc100 = cc100.map(
            preprocess_function,
            fn_kwargs={"tokenizer" : tokenizer},
            batched=True,
            num_proc=6,
            remove_columns=cc100.column_names
        )

    dataset = tokenized_cc100.map(group_texts, batched=True, num_proc=6)

    # Save preprocessed dataset to disk
    dataset.save_to_disk(f'preprocessed_cc100_{args.languages}')

    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--languages', type=str, default='nl')
    parser.add_argument('--model', type=str, default='xlm-roberta-base')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    prepare_dataset(args, tokenizer)