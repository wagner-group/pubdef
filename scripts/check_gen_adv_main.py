"""Script for checking generated adversarial examples."""

from __future__ import annotations

import sys
from pathlib import Path

from tabulate import tabulate

sys.path.append(".")
from exp_params import RESULT_PATH

# DATASET = "imagenet"
DATASET = "cifar100"


def main():
    """Main function."""
    result_path = Path(RESULT_PATH).expanduser()
    # dataset, model, attack, temperature, split, num_copies
    table = []
    for model_path in result_path.iterdir():
        if not model_path.is_dir():
            continue
        # Get dataset name
        dataset = model_path.name.split("_")[0]
        model = model_path.name.replace(f"{dataset}_", "")
        if dataset != DATASET:
            continue

        # Iterate over generated adversarial example dirs for this model
        for attack_path in model_path.iterdir():
            if not attack_path.is_dir():
                continue
            print(attack_path)
            saved, attack, split, temp = attack_path.name.split("_")[:4]
            assert saved == "saved"
            temp = float(temp.replace("temp", ""))
            temp = f"{temp:.2f}"

            # Get number of copies
            # Get first file in dir
            img_id = None
            for image_path in attack_path.glob("*/*.png"):
                img_id = image_path.stem.split("_")[0]
                break
            if img_id is None:
                continue
            # Get number of copies based on that one file alone
            copy_id = -1
            for image_path in attack_path.glob(f"*/{img_id}_*.png"):
                cur_copy_id = int(image_path.stem.split("_")[1])
                if cur_copy_id > copy_id:
                    copy_id = cur_copy_id
            num_copies = copy_id + 1

            row = [dataset, model, attack, temp, split, num_copies]
            table.append(row)
    table.sort()
    table = tabulate(
        table,
        headers=[
            "Dataset",
            "Model",
            "Attack",
            "Temp.",
            "Split",
            "Num. Copies",
        ],
    )
    with open("check_gen_adv_output.txt", "w", encoding="utf-8") as file:
        file.write(table)


if __name__ == "__main__":
    main()
