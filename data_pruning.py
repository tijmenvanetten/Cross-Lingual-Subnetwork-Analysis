import itertools

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


def preprocess_function(examples, tokenizer):
    # return tokenizer([" ".join(x) for x in examples["text"]])
    return tokenizer(examples["text"])


def group_texts(examples):
    block_size = 128

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


def prepare_dataset(args, tokenizer):
    languages = args.languages
    if isinstance(languages, str):
        languages = [languages]
    print("Setting seed to {}".format(args.seed))
    np.random.seed(args.seed)  # Set seed for random choice

    num_samples = args.train_samples + args.eval_samples
    train_sets, eval_sets = [], []
    for lang in languages:
        print("Loading {}...".format(lang))
        dataset_stream = load_dataset(
            "cc100", lang=lang, split="train", streaming=True
        )

        start_index = (args.seed - 1) * num_samples
        end_index = args.seed * num_samples

        # Use itertools.islice to select a slice of samples
        chosen_samples = list(
            itertools.islice(dataset_stream, start_index, end_index)
        )

        train_sets += chosen_samples[: args.train_samples]
        eval_sets += chosen_samples[args.train_samples : num_samples]
    cc100 = DatasetDict(
        {
            "train": Dataset.from_pandas(pd.DataFrame(data=train_sets)),
            "eval": Dataset.from_pandas(pd.DataFrame(data=eval_sets)),
        }
    ).shuffle(
        args.seed
    )  # Use the seed to shuffle

    print("Tokenizing...")
    tokenized_cc100 = cc100.map(
        preprocess_function,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        num_proc=4,
        remove_columns=cc100["train"].column_names,
    )

    print("Grouping texts...")
    dataset = tokenized_cc100.map(group_texts, batched=True, num_proc=6)

    print("Done w preparing dataset")
    return dataset