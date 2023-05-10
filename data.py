"""
Dataset: https://huggingface.co/datasets/cc100
Based on: https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling
"""
from datasets import load_dataset, concatenate_datasets, DatasetDict

def preprocess_function(examples, tokenizer):
    # return tokenizer([" ".join(x) for x in examples["text"]])
    return tokenizer(examples['text'])

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
    
    cc100_langs = []
    for lang in languages:
        cc100_lang = load_dataset("cc100", lang=lang, split="train").shuffle(seed=42).select(range(args.num_samples))
        cc100_langs.append(cc100_lang)

    cc100 = concatenate_datasets(cc100_langs)
    train_testvalid = cc100.train_test_split(test_size=1-args.train_split)
    # split the testalid in half test, half valid
    test_valid = train_testvalid["test"].train_test_split(test_size=0.5)

    cc100 = DatasetDict({
        "train": train_testvalid["train"],
        "eval": test_valid["train"],
        "test": test_valid["test"]})

    tokenized_cc100 = cc100.map(
            preprocess_function,
            fn_kwargs={"tokenizer" : tokenizer},
            batched=True,
            num_proc=6,
            remove_columns=cc100['train'].column_names
        )

    dataset = tokenized_cc100.map(group_texts, batched=True, num_proc=6)

    return dataset

