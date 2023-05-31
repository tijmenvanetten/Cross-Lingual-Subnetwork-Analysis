"""
Dataset: https://huggingface.co/datasets/cc100
Based on: https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling
"""

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
import random 

from language_props import lang_to_typology_dict


def preprocess_function(examples, tokenizer):
    return tokenizer(examples['text'])

def group_texts_lm(examples):
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

def prepare_lm_dataset(args, tokenizer):
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
        group_texts_lm,
        batched=True, 
        num_proc=6)

    return dataset

# def group_texts_typo(examples):
#     block_size = 128

#     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys() if isinstance(examples[k][0], list)}
#     print(concatenated_examples)
#     # Compute length of concatenated texts
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     print(total_length)
#     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#     # customize this part to your needs.
#     if total_length >= block_size:
#         total_length = (total_length // block_size) * block_size
#     # Split by chunks of block_size.
#     result = {
#         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
#         for k, t in concatenated_examples.items()
#     }
#     result['labels'] = examples['labels']
#     print(result)
#     return result 


def preprocess_typology_features(example, language, feature):
    example['label'] = lang_to_typology_dict[language][feature]
    return example

def preprocess_function_typo(examples, tokenizer):
    return tokenizer(examples['text'], padding=True, truncation=True)

def collect_langs(langs, num_samples, feature):
    lang_sets = []
    for lang in langs:
        samples = list(load_dataset("cc100", lang=lang, split="train", streaming=True).take(num_samples).map(
            preprocess_typology_features,
            fn_kwargs={"language": lang, "feature": feature},
        ))
        lang_sets += samples
    return lang_sets

def prepare_typology_dataset(args, tokenizer):
    train_langs = [args.train_langs] if isinstance(args.train_langs, str) else args.train_langs
    eval_langs = [args.eval_langs] if isinstance(args.eval_langs, str) else args.eval_langs

    train_sets = collect_langs(train_langs, args.train_samples, args.feature)
    held_out_sets = collect_langs(eval_langs, args.eval_samples + args.test_samples, args.feature)
    random.shuffle(held_out_sets)

    eval_idx = args.eval_samples * len(eval_langs)
    eval_sets, test_sets = held_out_sets[:eval_idx], held_out_sets[eval_idx:]

    cc100 = DatasetDict({
        "train": Dataset.from_pandas(pd.DataFrame(data=train_sets)),
        "eval": Dataset.from_pandas(pd.DataFrame(data=eval_sets)),
        "test": Dataset.from_pandas(pd.DataFrame(data=test_sets)),
        }).shuffle(42)
    
    tokenized_cc100 = cc100.map(
            preprocess_function_typo,
            fn_kwargs={"tokenizer" : tokenizer},
            batched=True,
            batch_size=32,
            num_proc=6,
            remove_columns=['id', 'text']
        )

    # dataset = tokenized_cc100.rename_column("input_ids", "text")

    return tokenized_cc100

def preprocess_function_rsa(examples, tokenizer):
    return tokenizer(examples['0'], padding=True, truncation=True)

def prepare_rsa_dataset(args, tokenizer):
    lang1 = []
    lang2 = []
    if len(args.compare_languages) == 1:
        samples = list(load_dataset("tatoeba", lang1='en', lang2=args.compare_languages[0], split=f'train[:{args.test_samples}]'))
        for elem in samples:
            lang1.append(elem['translation'][args.compare_languages[0]])

        tatoeba = DatasetDict({
            "lang1": Dataset.from_pandas(pd.DataFrame(data=lang1)),
            "lang2": Dataset.from_pandas(pd.DataFrame(data=lang1)),
            })
    else:
        lang1 = []
        lang2 = []
        
        if 'sw' in args.compare_languages:
            args.compare_languages = list(args.compare_languages)
            idx = args.compare_languages.index('sw')
            args.compare_languages[idx] = 'swh'
        try:
            print('Language pair:', args.compare_languages[0], args.compare_languages[1])
            samples = list(load_dataset("tatoeba", lang1=args.compare_languages[0], lang2=args.compare_languages[1], split=f'train[:{args.test_samples}]'))
        except FileNotFoundError:
            print('Language pair:', args.compare_languages[1], args.compare_languages[0])
            samples = list(load_dataset("tatoeba", lang1=args.compare_languages[1], lang2=args.compare_languages[0], split=f'train[:{args.test_samples}]'))
        except ValueError:
            print('Trying again with the same configuration but with the whole dataset')
            samples = list(load_dataset("tatoeba", lang1=args.compare_languages[0], lang2=args.compare_languages[1], split=f'train'))
        for elem in samples:
            lang1.append(elem['translation'][args.compare_languages[0]])
            lang2.append(elem['translation'][args.compare_languages[1]])

        tatoeba = DatasetDict({
            "lang1": Dataset.from_pandas(pd.DataFrame(data=lang1)),
            "lang2": Dataset.from_pandas(pd.DataFrame(data=lang2)),
            })

    dataset = tatoeba.map(
            preprocess_function_rsa,
            fn_kwargs={"tokenizer" : tokenizer},
            batched=True,
            num_proc=6,
            remove_columns=['0']
        )

    return dataset

