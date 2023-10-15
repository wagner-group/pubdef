"""Randomly subsample the ImageNet dataset."""

import random
from pathlib import Path

import numpy as np


def select_random_samples(
    base_path, num_samples_per_class, output_file, seed=42
):
    """Randomly select a given number of samples from each class in ImageNet.

    Args.:
        base_path (Path): The base directory containing class-wise subdirectories.
        num_samples_per_class (int): The number of samples to select per class.
        output_file (str): The name of the output text file where selected file names will be saved.
        seed (int): Random seed for reproducibility.
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Check if the base path exists
    if not base_path.exists():
        print("The specified base path does not exist.")
        return

    # Initialize an empty list to store selected file names
    selected_files = []

    # Loop through each sub-directory (class)
    for class_path in base_path.iterdir():
        if class_path.is_dir():
            # List all files in the sub-directory
            all_files = list(class_path.glob("*.jpg"))

            # Randomly select 'num_samples_per_class' files
            if len(all_files) >= num_samples_per_class:
                selected = random.sample(all_files, num_samples_per_class)
            else:
                print(
                    f"Class {class_path.name} has fewer samples ({len(all_files)}) "
                    f"than requested ({num_samples_per_class}). Selecting all."
                )
                selected = all_files

            # Append selected files to the list
            selected_files.extend(
                [str(s.relative_to(base_path)) for s in selected]
            )

    # Save the selected file names to a text file
    with open(output_file, "w", encoding="utf-8") as file:
        for file_name in selected_files:
            file.write(f"{file_name}\n")

    print(f"Selected samples have been saved to {output_file}.")


if __name__ == "__main__":
    # Change this to the path where your ImageNet dataset is stored
    BASE_PATH = Path("~/data/imagenet/val/").expanduser()

    # Change this to specify the number of random samples you want per class
    NUM_SAMPLES_PER_CLASS = 10

    # The output text file name
    OUTPUT_FILE = "selected_samples.txt"

    # Random seed for reproducibility
    SEED = 0

    select_random_samples(BASE_PATH, NUM_SAMPLES_PER_CLASS, OUTPUT_FILE, SEED)
