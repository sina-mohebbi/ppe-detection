"""
Prepare the PPE dataset for training.

The raw dataset ships as a single PPE.zip (YOLO format: train/valid/test folders
with images + labels, plus a data.yaml). This script:

  1. Extracts the zip into data/
  2. Locates the data.yaml
  3. Rewrites its paths to ABSOLUTE paths, so Ultralytics finds the images no
     matter which directory you launch training from (a very common source of
     "0 images found" headaches).
  4. Prints a quick sanity summary: image counts per split and the class names.

Run once, after PPE.zip has finished downloading:
    python src/prepare_data.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import yaml

# Project layout ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]        # project root
DATA_DIR = ROOT / "data"
ZIP_PATH = DATA_DIR / "PPE.zip"


def extract_zip() -> None:
    """Unzip the dataset into data/ (skips if it looks already extracted)."""
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"{ZIP_PATH} not found. Download it first:\n"
            "  curl.exe -L -o data/PPE.zip "
            "https://huggingface.co/datasets/51ddhesh/PPE_Detection/resolve/main/PPE.zip"
        )

    print(f"Extracting {ZIP_PATH.name} ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)
    print("  done.")


def find_data_yaml() -> Path:
    """Find the dataset's data.yaml wherever the zip placed it."""
    candidates = list(DATA_DIR.rglob("data.yaml"))
    if not candidates:
        raise FileNotFoundError("No data.yaml found under data/ after extraction.")
    # Prefer the shallowest one if several exist.
    return min(candidates, key=lambda p: len(p.parts))


def fix_paths(yaml_path: Path) -> dict:
    """Rewrite train/val/test to absolute paths and return the parsed config."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    base = yaml_path.parent  # paths in the yaml are relative to its own folder

    def resolve(split_value: str) -> str:
        # Handles both "train/images" and "../train/images" style entries.
        p = (base / split_value).resolve()
        if not p.exists():
            # Some exports point at the split folder; try common fallbacks.
            alt = (base / split_value.replace("../", "")).resolve()
            if alt.exists():
                p = alt
        return str(p)

    for split in ("train", "val", "test"):
        if split in cfg and cfg[split]:
            cfg[split] = resolve(cfg[split])

    # Drop any relative 'path' root now that splits are absolute.
    cfg.pop("path", None)

    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return cfg


def summarize(yaml_path: Path, cfg: dict) -> None:
    """Print image counts and classes so we know the data is sane."""
    print("\n" + "=" * 55)
    print(f"data.yaml: {yaml_path}")
    names = cfg.get("names")
    print(f"classes ({len(names)}): {names}")
    for split in ("train", "val", "test"):
        folder = cfg.get(split)
        if folder and Path(folder).exists():
            n = sum(1 for _ in Path(folder).glob("*.*"))
            print(f"  {split:5s}: {n:5d} images  ({folder})")
    print("=" * 55)
    # Stash the resolved yaml path for other scripts to read.
    (ROOT / "data" / "ACTIVE_DATA_YAML.txt").write_text(str(yaml_path))
    print(f"\nWrote pointer -> data/ACTIVE_DATA_YAML.txt")


if __name__ == "__main__":
    extract_zip()
    yml = find_data_yaml()
    config = fix_paths(yml)
    summarize(yml, config)
    print("\nDataset ready. Next: python src/train.py")
