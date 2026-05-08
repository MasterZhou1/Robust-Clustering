#!/usr/bin/env python3
"""Download raw datasets used by the benchmarks."""

import os
import gzip
import shutil
import urllib.request
from pathlib import Path

DATASET_DIR = Path(__file__).parent

DATASETS = {
    "skin": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt",
        "filename": "Skin_NonSkin.txt",
        "description": "Skin Segmentation",
    },
    "susy": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz",
        "filename": "SUSY.csv.gz",
        "description": "SUSY",
    },
    "shuttle": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.trn.Z",
        "filename": "shuttle.trn.Z",
        "description": "Shuttle (training split)",
    },
    "shuttle_test": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.tst",
        "filename": "shuttle.tst",
        "description": "Shuttle (test split)",
    },
    "kddfull": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data.gz",
        "filename": "kddcup.data.gz",
        "description": "KDD Cup 1999",
    },
}


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file from URL to destination path."""
    if dest_path.exists():
        print(f"  [SKIP] {dest_path.name} already exists")
        return True
    
    print(f"  Downloading from: {url}")
    try:
        # Create a custom opener with a user agent
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, dest_path)
        print(f"  [OK] Saved to: {dest_path}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to download: {e}")
        return False


def decompress_gz(gz_path: Path) -> Path:
    """Decompress a .gz file."""
    output_path = gz_path.with_suffix('')
    if output_path.exists():
        print(f"  [SKIP] {output_path.name} already decompressed")
        return output_path
    
    print(f"  Decompressing {gz_path.name}...")
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"  [OK] Decompressed to: {output_path.name}")
        return output_path
    except Exception as e:
        print(f"  [ERROR] Failed to decompress: {e}")
        return None


def decompress_z(z_path: Path) -> Path:
    """Decompress a .Z file using system uncompress or gzip."""
    output_path = z_path.with_suffix('')
    if output_path.exists():
        print(f"  [SKIP] {output_path.name} already decompressed")
        return output_path
    
    print(f"  Decompressing {z_path.name}...")
    try:
        # Try using gzip (most systems have this)
        import subprocess
        result = subprocess.run(['gzip', '-d', '-k', str(z_path)], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  [OK] Decompressed to: {output_path.name}")
            return output_path
        else:
            # Try uncompress
            result = subprocess.run(['uncompress', '-k', str(z_path)],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  [OK] Decompressed to: {output_path.name}")
                return output_path
            print(f"  [ERROR] Failed to decompress .Z file: {result.stderr}")
            return None
    except Exception as e:
        print(f"  [ERROR] Failed to decompress: {e}")
        return None


def main():
    print("=" * 60)
    print("Downloading Clustering Datasets")
    print("=" * 60)
    
    for name, info in DATASETS.items():
        print(f"\n[{name.upper()}] {info['description']}")
        print("-" * 40)
        
        dest_path = DATASET_DIR / info['filename']
        
        if download_file(info['url'], dest_path):
            # Decompress if needed
            if info['filename'].endswith('.gz'):
                decompress_gz(dest_path)
            elif info['filename'].endswith('.Z'):
                decompress_z(dest_path)
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    
    # Show downloaded files
    print("\nDownloaded files:")
    for f in sorted(DATASET_DIR.glob('*')):
        if f.is_file() and f.name != 'download_datasets.py':
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
