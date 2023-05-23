from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import argparse
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

def preprocess_function(examples, tokenizer):
    return tokenizer(examples)

def padding(examples):
    return pad_sequence(examples, batch_first=True, padding_value=0)

def create_reps(args, model, dataset):
    # arguments for Trainer
    test_args = TrainingArguments(
        output_dir = args.output_dir,
        do_train = False,
        do_predict = True,
        per_device_eval_batch_size = args.batch_size,   
        dataloader_drop_last = False    
    )

    # init trainer
    trainer = Trainer(
                model = model, 
                args = test_args)

    test_results = trainer.predict(dataset['lang1'])

    return test_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--languages', nargs="*", default=['en', 'mr'],
                        help='Language to finetune on.')
    parser.add_argument('--tokenizer', default="xlm-roberta-base", type=str,
                       help='Tokenizer used.')
    parser.add_argument('--model_checkpoint', default='xlm-roberta-base', type=str,
                        help='Path to checkpoints of subnetworks.')
    parser.add_argument('--test_samples', default=50, type=int,
                       help='Number of test samples.')
    parser.add_argument('--batch_size', default=28, type=int,
                        help='Batch size')
    parser.add_argument('--output_dir', default='./', type=str)
    
    args = parser.parse_args()

    dataset = load_dataset("tatoeba", lang1=args.languages[0], lang2=args.languages[1], split=f'train[:{args.test_samples}]')

    lang1 = []
    lang2 = []
    for item in dataset:
        lang1.append(item['translation'][args.languages[0]])
        lang2.append(item['translation'][args.languages[1]])

    tatoeba = DatasetDict({
    "lang1": Dataset.from_pandas(pd.DataFrame(data=lang1)),
    "lang2": Dataset.from_pandas(pd.DataFrame(data=lang2)),
    })

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # change to args.model if pruned models are known
    model = AutoModelForMaskedLM.from_pretrained(args.tokenizer)
    print(tatoeba)
    tokenized_tatoeba = tatoeba.map(
            preprocess_function,
            fn_kwargs={"tokenizer" : tokenizer},
            batched=True,
            batch_size=28,
            num_proc=1
        )
    
    tatoe_dataset = tokenized_tatoeba.map(
        padding,
        num_proc=1)
    
    print(create_reps(model, tokenizer, tokenized_tatoeba))
    # encodings = torch.vstack(create_reps(model, tokenizer, language1), create_reps(model, tokenizer, language2))

    """
    Stappenplan:
    1. Preprocess dataset zodat het in het model kan.
    2. Vergelijk outputs (encoding) based on cosine similarity.
    3. stop het in een matrix.
    """







    




