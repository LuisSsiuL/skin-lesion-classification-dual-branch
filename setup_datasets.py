#!/usr/bin/env python3
"""
Download Derm7pt and MILK10k datasets from Kaggle and place them where
model.ipynb expects them:

  dataset/Derm7pt/meta/meta.csv
  dataset/Derm7pt/images/...
  dataset/milk10k/MILK10k_Training_Metadata.csv
  dataset/milk10k/MILK10k_Training_GroundTruth.csv
  dataset/milk10k/MILK10k_Training_Input/...

Prerequisites
-------------
1. A Kaggle account: https://www.kaggle.com
2. A Kaggle API token at  ~/.kaggle/kaggle.json
   (Kaggle → Settings → API → "Create New Token")
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

PROJECT = Path(__file__).parent
DATASET = PROJECT / "dataset"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pip_install(package: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)


def ensure_kaggle() -> None:
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("kaggle CLI not found — installing…")
        pip_install("kaggle")


def check_api_key() -> None:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print()
        print("=" * 60)
        print("Kaggle API key not found.")
        print()
        print("Steps to get one:")
        print("  1. Go to https://www.kaggle.com/settings/account")
        print("  2. Scroll to the 'API' section")
        print("  3. Click 'Create New Token' — kaggle.json downloads")
        print(f"  4. Move it to: {kaggle_json}")
        print("     (create the .kaggle folder if it doesn't exist)")
        print()
        print("Then re-run this script.")
        print("=" * 60)
        sys.exit(1)
    # restrict permissions on non-Windows systems
    if os.name != "nt":
        kaggle_json.chmod(0o600)


def download_dataset(dataset_id: str, dest: Path, sentinel: str) -> None:
    """
    Download a Kaggle dataset, extract it, and place the contents in *dest*.

    Handles the case where Kaggle wraps everything in a top-level subfolder.
    Skips the download if *sentinel* already exists inside *dest*.
    """
    if (dest / sentinel).exists():
        print(f"  Already present — skipping ({dest.name})")
        return

    tmp = DATASET / f"_tmp_{dest.name}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {dataset_id} …")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(tmp), "--unzip"],
        check=True,
    )

    # Some zips wrap everything in a single top-level folder; detect and flatten.
    items = [p for p in tmp.iterdir()]
    if len(items) == 1 and items[0].is_dir():
        src = items[0]
    else:
        src = tmp

    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dest / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), str(target))

    shutil.rmtree(tmp, ignore_errors=True)

    if (dest / sentinel).exists():
        print(f"  OK — {dest.name}")
    else:
        print(f"\n  WARNING: expected file not found after extraction:")
        print(f"    {dest / sentinel}")
        print(f"  Actual contents of {dest}:")
        for p in sorted(dest.rglob("*"))[:30]:
            print(f"    {p.relative_to(dest)}")
        print()
        print("  The dataset structure may have changed. Check the paths in")
        print("  model.ipynb (DERM7PT_DIR / MILK_DIR) and adjust as needed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_kaggle()
    check_api_key()
    DATASET.mkdir(exist_ok=True)

    print("\n── Derm7pt ─────────────────────────────────────────────────────")
    download_dataset(
        dataset_id="menakamohanakumar/derm7pt",
        dest=DATASET / "Derm7pt",
        sentinel="meta/meta.csv",
    )

    print("\n── MILK10k ──────────────────────────────────────────────────────")
    download_dataset(
        dataset_id="nguyenphucduyloc/milk10k-isic-challenge-2025",
        dest=DATASET / "milk10k",
        sentinel="MILK10k_Training_Metadata.csv",
    )

    print("\n────────────────────────────────────────────────────────────────")
    print("Done. Open model.ipynb and run cells sequentially.")


if __name__ == "__main__":
    main()
