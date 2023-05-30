import argparse
import os
import torch
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

from data import prepare_rsa_dataset
from tqdm import tqdm
from masking import prune_model

from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import seaborn as sns

def RDM(reps):
    # p is Minkowski p-norm 
    dist_matrices = []
    for embed in reps:
        dist_mat = distance_matrix(embed, embed, p=2)
        dist_matrices.append(dist_mat)
    return dist_matrices

def create_representations(args, model, trainer):
    dataloader = trainer.get_train_dataloader()
    
    model = model.to(args.device)
    model.eval()
    print(len(dataloader))
    for idx, batch in enumerate(dataloader):
        batch = batch.to(args.device)
        input_ids, input_mask, label_ids = batch['input_ids'], batch['attention_mask'], batch['labels']

        with torch.no_grad():
            # outputs = (loss, logits, hidden_states) --> mean-pooling last hiddenstate
            outputs = torch.mean(model(
                input_ids, attention_mask=input_mask, labels=label_ids, head_mask=None,
                output_hidden_states=True
            )[2][-1], 1)

        if idx == 0:
            reps = outputs
        else:
            reps = torch.vstack((reps, outputs))

    return reps

def initialize_rsa(args, model1, model2, tokenizer, data_collator):
    """Evaluate the model for each language specific mask on each language."""
    print("Initializing model and data for RSA...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    training_args = TrainingArguments(
        output_dir=f'results_rsa',
        overwrite_output_dir = 'False',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        num_train_epochs=10,
        weight_decay=0.01,
        push_to_hub=False,
        save_total_limit = 2,
        save_strategy = 'no',
        load_best_model_at_end=False
    )

    rdms = []
    dataset = prepare_rsa_dataset(args, tokenizer)

    models = (model1, model2)

    for idx, lang_data in enumerate([dataset['lang1'], dataset['lang2']]):
        trainer = Trainer(
            model=models[idx],
            args=training_args,
            train_dataset=lang_data,
            data_collator=data_collator
        )

        reps = create_representations(args, model, trainer)
        rdms.append(reps)
    
    return rdms

def plot_RSA(args, rsa_matrix):
    ax = sns.heatmap(rsa_matrix, linewidth=0.5)
    
    if len(args.compare_languages) == 1:
        plt.title(f'Similarity full- and subnetwork for {args.compare_languages[0]}')
        plt.savefig(f'./results_rsa/rsa_full_sub_{args.compare_languages[0]}.png')
    else:
        plt.title(f'Similarity between {args.compare_languages[0]} and {args.compare_languages[1]}')
        plt.savefig(f'./results_rsa/rsa_2lang_{args.compare_languages[0]}_{args.compare_languages[1]}.png')

    plt.clf()

def RSA(distances):
    return cdist(distances[0], distances[1], 'cosine')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks_dir", default="masks/head_masks", type=str,
                        help="Directory containing the masks")
    parser.add_argument("--model", default="xlm-roberta-base", type=str,
                        help="Model to use")
    parser.add_argument('--test_split', default=0.2, type=float,
                        help='Percentage of test split.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed.')
    parser.add_argument('--test_samples', default=20, type=int,
                        help='Number of training samples.')
    parser.add_argument('--compare_languages', nargs="*", default=['en'],
                        help='Languages to evaluate on.')
    parser.add_argument('--checkpoint', default="/Users/sperdijk/Documents/Master/Jaar 2/logs/results_['en', 'nl', 'fy', 'he', 'ar', 'hi', 'ur', 'sw', 'zu', 'cy', 'gd']/", type=str,
                       help='Pretrained encoder to use.')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    model = AutoModelForMaskedLM.from_pretrained(args.checkpoint)
    # 'ar', 'cy', 'en' fy', 'gd', 'he', 'hi', 'nl', 
    # poss_lang = ['sw', 'ur', 'zu']
    # niet hi & ar en niet he swh en niet hi nl
    # combinations =  [('ar', 'en'), ('ar', 'fy'), ('ar', 'he'), ('ar', 'hi'), ('ar', 'nl'), ('ar', 'ur'),
    #                  ('cy', 'en'), ('cy', 'gd'), ('cy', 'he'),
    #                  ('en', 'fy'), ('en', 'gd'), ('en', 'he'), ('en', 'hi'), ('en', 'nl'), ('en', 'sw'), ('en', 'ur'), ('en', 'zu'),
    #                  ('fy', 'he'), ('fy', 'nl'), ('fy', 'sw'), 
    #                  ('gd', 'nl'),
    #                  ('he', 'nl'), ('he', 'sw'), 
    #                  ('hi', 'nl'), ('hi', 'ur'),
    #                  ('nl', 'sw')
    #                  ]

    combinations =  [
                     ('hi', 'ur'),
                     ('nl', 'sw')
                     ]
    for i in range(len(combinations)):
        args.compare_languages = combinations[i]
        if len(args.compare_languages) != 1:
            # Compare multiple languages with their subnetworks
            args.mask = os.path.join(args.masks_dir, f"average_mask_{args.compare_languages[0]}.npy")
            model1 = prune_model(args, model)

            args.mask = os.path.join(args.masks_dir, f"average_mask_{args.compare_languages[1]}.npy")
            model2 = prune_model(args, model)
        else:
            # Compare subnetwork with full network 
            args.mask = os.path.join(args.masks_dir, f"average_mask_{args.compare_languages[0]}.npy")
            print('Masked used: ', args.mask)
            model2 = prune_model(args, model)
            model1 = model

        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

        # Get representations of full model / subnetwork or lang1 / lang2
        reps = initialize_rsa(args, model1, model2, tokenizer, data_collator)

        # Create two RDMS with distances
        distances = RDM(reps)

        # Compare RDMS with RSA
        rsa_matrix = RSA(distances)
        plot_RSA(args, rsa_matrix)

        del model2
        del model1
