"""
Robust Dataset Ingestion & Extraction for Windows.
Bypasses the Windows MAX_PATH (260-character) limitation by:
1. Stripping redundant nested Kaggle folder prefixes on the fly.
2. Directly routing files to dataset/train and dataset/valid.
3. Using Windows extended-length path prefix (\\\\?\\).
"""

import os
import sys
import zipfile
import shutil
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

def make_long_path(path: str) -> str:
    """Wraps path with Windows extended-length prefix to eliminate 260-char limit."""
    abs_path = os.path.abspath(path)
    if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
        return '\\\\?\\' + abs_path
    return abs_path

def setup_dataset_from_zip(zip_path: str):
    if not os.path.exists(zip_path):
        print(f"❌ Error: Zip file not found at: {zip_path}")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)
    print("=" * 65)
    print(f"📦 Source Zip: {zip_path}")
    print(f"📂 Target Directory: {DATASET_DIR}")
    print("=" * 65)

    print("Analyzing zip archive structure...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        total_files = len([m for m in members if not m.is_dir()])
        print(f"Found {total_files:,} files inside archive.")

        extracted_count = 0
        train_count = 0
        val_count = 0
        test_count = 0
        start_time = time.time()

        for member in members:
            if member.is_dir():
                continue

            # Normalize path
            norm_name = member.filename.replace('\\', '/')
            parts = norm_name.strip('/').split('/')

            target_rel = None
            if 'train' in parts:
                idx = parts.index('train')
                if len(parts) >= idx + 3:
                    class_name = parts[idx + 1]
                    img_name = parts[-1]
                    target_rel = os.path.join('train', class_name, img_name)
                    train_count += 1
            elif 'valid' in parts or 'val' in parts:
                idx = parts.index('valid') if 'valid' in parts else parts.index('val')
                if len(parts) >= idx + 3:
                    class_name = parts[idx + 1]
                    img_name = parts[-1]
                    target_rel = os.path.join('valid', class_name, img_name)
                    val_count += 1
            elif 'test' in parts:
                idx = parts.index('test')
                if len(parts) >= idx + 2:
                    sub_parts = parts[idx + 1:]
                    target_rel = os.path.join('test', *sub_parts)
                    test_count += 1

            # Fallback: if not in standard train/valid structure, extract under dataset/
            if target_rel is None:
                continue

            target_full = os.path.join(DATASET_DIR, target_rel)
            target_dir = os.path.dirname(target_full)

            # Ensure parent directory exists using long path
            os.makedirs(make_long_path(target_dir), exist_ok=True)

            # Extract file stream
            with zip_ref.open(member) as source, open(make_long_path(target_full), 'wb') as dest:
                shutil.copyfileobj(source, dest)

            extracted_count += 1

            if extracted_count % 5000 == 0 or extracted_count == total_files:
                elapsed = time.time() - start_time
                percent = (extracted_count / total_files) * 100
                speed = extracted_count / (elapsed + 1e-6)
                print(f"⏳ Extracted {extracted_count:,} / {total_files:,} images ({percent:.1f}%) | {speed:.0f} img/s")

    elapsed_total = time.time() - start_time
    print("=" * 65)
    print("✅ Extraction Complete!")
    print(f"📊 Total Extracted: {extracted_count:,} images in {elapsed_total:.1f}s")
    print(f"  ├── Train Images: {train_count:,} (in dataset/train)")
    print(f"  ├── Valid Images: {val_count:,} (in dataset/valid)")
    if test_count > 0:
        print(f"  └── Test Images:  {test_count:,} (in dataset/test)")
    print("=" * 65)
    print("🚀 You can now start training immediately:")
    print("   python training/train_pytorch.py --train_dir dataset/train --val_dir dataset/valid --batch_size 16")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        setup_dataset_from_zip(sys.argv[1])
    else:
        print("Usage: python data/download_dataset.py \"C:\\path\\to\\archive.zip\"")
