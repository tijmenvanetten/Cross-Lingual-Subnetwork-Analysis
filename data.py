"""
Dataset: https://huggingface.co/datasets/cc100
Based on: https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling
"""

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from language_props import lang_to_typology_dict

def preprocess_typology_features(example, language):
    example['word_order'] = lang_to_typology_dict[language]['word_order']
    example['writing_system'] = lang_to_typology_dict[language]['writing_system']
    return example

def preprocess_function(examples, tokenizer):
    return tokenizer(examples['text'])

def group_texts(examples):
    block_size = 128

    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys() }

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

def prepare_dataset(args, tokenizer):
    languages = [args.languages] if isinstance(args.languages, str) else args.languages

    num_samples = args.train_samples + args.eval_samples
    train_sets, eval_sets = [], []
    for lang in languages:
        samples = list(load_dataset("cc100", lang=lang, split="train", streaming=True).take(num_samples))
        
        train_sets += samples[:args.train_samples]
        eval_sets += samples[-args.eval_samples:]

    cc100 = DatasetDict({
        "train": Dataset.from_pandas(pd.DataFrame(data=train_sets)),
        "eval": Dataset.from_pandas(pd.DataFrame(data=eval_sets)),
        }).shuffle(42)
    
    tokenized_cc100 = cc100.map(
            preprocess_function,
            fn_kwargs={"tokenizer" : tokenizer},
            batched=True,
            num_proc=6,
            remove_columns=['id', 'text']
        )

    dataset = tokenized_cc100.map(
        group_texts,
        batched=True, 
        num_proc=6)

    return dataset

def collect_langs(langs, samples):
    lang_sets = []
    for lang in langs:
        samples = list(load_dataset("cc100", lang=lang, split="train", streaming=True).take(samples).map(
            preprocess_typology_features,
            fn_kwargs={"language": lang},
        ))
        
        lang_sets += samples
    return lang_sets

def prepare_typology_dataset(args, tokenizer):
    train_langs = [args.train_langs] if isinstance(args.train_langs, str) else args.train_langs
    eval_langs = [args.eval_langs] if isinstance(args.eval_langs, str) else args.eval_langs

    train_sets = collect_langs(train_langs, args.train_samples)
    eval_sets = collect_langs(eval_langs, args.eval_samples)

    cc100 = DatasetDict({
        "train": Dataset.from_pandas(pd.DataFrame(data=train_sets)),
        "eval": Dataset.from_pandas(pd.DataFrame(data=eval_sets)),
        }).shuffle(42)
    
    tokenized_cc100 = cc100.map(
            preprocess_function,
            fn_kwargs={"tokenizer" : tokenizer},
            batched=True,
            num_proc=6,
            remove_columns=['id', 'text']
        )

    dataset = tokenized_cc100.map(
        group_texts,
        batched=True, 
        num_proc=6)

    return dataset

