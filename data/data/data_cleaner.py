"""
Data Cleaning Utility:
Scans the dataset directory to detect and sanitize:
1. Truncated or corrupt image files (using PIL.Image.verify()).
2. Zero-byte or non-image files.
3. MD5/SHA256 duplicate image detection across classes.
"""

import os
import hashlib
from typing import List, Dict, Set
from PIL import Image

VALID_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def verify_and_clean_images(data_dir: str, remove_corrupt: bool = False) -> Dict[str, List[str]]:
    """
    Scans data directory recursively, verifies images, and detects corrupt or duplicate files.
    """
    corrupt_files: List[str] = []
    zero_byte_files: List[str] = []
    hash_map: Dict[str, str] = {}
    duplicate_files: List[str] = []
    total_scanned = 0

    print(f"Scanning directory: {data_dir}")
    for root, _, files in os.walk(data_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue

            file_path = os.path.join(root, file)
            total_scanned += 1

            # 1. Check file size
            try:
                size = os.path.getsize(file_path)
                if size == 0:
                    zero_byte_files.append(file_path)
                    if remove_corrupt:
                        os.remove(file_path)
                    continue
            except OSError as e:
                corrupt_files.append(file_path)
                continue

            # 2. Verify image integrity with PIL
            try:
                with Image.open(file_path) as img:
                    img.verify()
                # Reopen to test load (verify doesn't decode pixels)
                with Image.open(file_path) as img:
                    img.load()
            except Exception as e:
                corrupt_files.append(file_path)
                if remove_corrupt:
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass
                continue

            # 3. Detect exact duplicates via hash
            try:
                with open(file_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash in hash_map:
                    duplicate_files.append(file_path)
                else:
                    hash_map[file_hash] = file_path
            except Exception:
                pass

    print(f"Total scanned images: {total_scanned}")
    print(f"Corrupt/Unreadable images: {len(corrupt_files)}")
    print(f"Zero-byte images: {len(zero_byte_files)}")
    print(f"Duplicate images: {len(duplicate_files)}")

    return {
        "corrupt": corrupt_files,
        "zero_byte": zero_byte_files,
        "duplicates": duplicate_files
    }

if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    verify_and_clean_images(target_dir, remove_corrupt=False)
