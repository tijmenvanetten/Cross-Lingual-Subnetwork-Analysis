import argparse
import os

import numpy as np


def calculate_average_mask(masks):
    average_mask = np.sum(masks, axis=0) / 3
    # If value above 0.5, set to 1, else set to 0
    average_mask[average_mask > 0.5] = 1
    average_mask[average_mask <= 0.5] = 0

    return average_mask

def main():
    # Get the list of languages
    mask_files = os.listdir(args.masks_dir)
    print(f"Found masks: {mask_files}")
    languages = set([lang[10:12] for lang in mask_files])

    # For each language, load all masks and calculate the average mask
    for language in languages:
        masks = []
        for mask_file in mask_files:
            if mask_file[10:12] == language:
                masks.append(np.load(os.path.join(args.masks_dir, mask_file)))
        
        average_mask = calculate_average_mask(np.array(masks))
        np.save(os.path.join(args.masks_dir, f"average_mask_{language}.npy"), average_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks_dir", type=str, default="head_masks", help="Directory containing the masks")
    args = parser.parse_args()

    main()