"""
Dataset Splitter Utility:
Creates a clean, stratified 3-way split (Train / Validation / Test)
from raw class folders to prevent data leakage.
"""

import os
import shutil
import random
from typing import Tuple

def create_stratified_splits(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> None:
    """
    Performs stratified partition of images across classes into train, val, and test splits.
    Ensures zero overlap and identical class balance across splits.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    random.seed(seed)

    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f"Found {len(classes)} classes in {source_dir}")

    splits = ['train', 'val', 'test']
    for s in splits:
        for c in classes:
            os.makedirs(os.path.join(output_dir, s, c), exist_ok=True)

    summary = {}
    for c in sorted(classes):
        class_path = os.path.join(source_dir, c)
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for img in train_imgs:
            shutil.copy2(os.path.join(class_path, img), os.path.join(output_dir, 'train', c, img))
        for img in val_imgs:
            shutil.copy2(os.path.join(class_path, img), os.path.join(output_dir, 'val', c, img))
        for img in test_imgs:
            shutil.copy2(os.path.join(class_path, img), os.path.join(output_dir, 'test', c, img))

        summary[c] = {'train': len(train_imgs), 'val': len(val_imgs), 'test': len(test_imgs)}

    print(f"Stratified partition complete. Output saved to {output_dir}")
    return summary

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        create_stratified_splits(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python dataset_splitter.py <source_dir> <output_dir>")
