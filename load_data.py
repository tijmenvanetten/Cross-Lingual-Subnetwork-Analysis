import datasets
from pathlib import Path
import argparse

def load(args):
    return datasets.load_dataset("cc100", lang=args.languages, split="train")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--languages', default='gn', type=str,
                        help='Language to finetune on.')
    parser.add_argument('--target_dir', default='./results_nl/', type=Path)
    
    args = parser.parse_args()
    datasets.config.DOWNLOADED_DATASETS_PATH = args.target_dir

    load(args)
