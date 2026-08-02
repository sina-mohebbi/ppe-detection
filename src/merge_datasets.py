"""
Merge an external YOLO dataset into ours, harmonizing the class taxonomy.

Different public datasets use different class names and orderings, so you can't
just dump one into another — the label indices would mean different things. This
script remaps a source dataset's class ids onto ours and drops classes we don't
use, then copies the images + rewritten labels into our train/valid folders.

Here it folds in the Ultralytics **Construction-PPE** dataset, which overlaps 5
of our 6 classes (and notably adds gloves data — our weakest class):

    source id / name        ->  our id / name
    0 helmet                ->  3 helmet
    1 gloves                ->  0 Gloves
    2 vest                  ->  1 Vest
    3 boots                 ->  5 safety_shoe   (imperfect match — see note below)
    4 goggles               ->  2 goggles
    5 none, 6 Person,
    7-10 no_* (violations)  ->  dropped

Test splits are kept separate so evaluation stays a fair, unchanged benchmark.

Usage (run once, after downloading construction-ppe):
    python src/merge_datasets.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "construction-ppe"
OURS = ROOT / "data"

# source class id -> our class id  (ids absent here are dropped)
# note on "boots" (source id 3 -> safety_shoe): this mapping is imperfect (boots
# are not exactly safety shoes) and slightly lowers safety_shoe accuracy. I tested
# dropping it, which recovered safety_shoe by ~0.02 but hurt overall accuracy and
# cross-dataset generalization more — so it's kept. See reports/dataset_merge.md.
CLASS_MAP = {0: 3, 1: 0, 2: 1, 3: 5, 4: 2}
# source split folder -> our split folder
SPLIT_MAP = {"train": "train", "val": "valid"}
PREFIX = "cppe_"          # filename prefix to avoid collisions
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

OUR_NAMES = ["Gloves", "Vest", "goggles", "helmet", "mask", "safety_shoe"]


def find_image(stem: str, img_dir: Path) -> Path | None:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def remap_label(text: str) -> tuple[str, dict[int, int]]:
    """Return (rewritten label text, per-class counts kept)."""
    out_lines, counts = [], {}
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        src_id = int(float(parts[0]))
        if src_id not in CLASS_MAP:
            continue                       # drop classes we don't use
        dst_id = CLASS_MAP[src_id]
        out_lines.append(" ".join([str(dst_id)] + parts[1:]))
        counts[dst_id] = counts.get(dst_id, 0) + 1
    return "\n".join(out_lines), counts


def merge_split(src_split: str, dst_split: str) -> dict:
    src_img = SOURCE / "images" / src_split
    src_lbl = SOURCE / "labels" / src_split
    dst_img = OURS / dst_split / "images"
    dst_lbl = OURS / dst_split / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    added, skipped, totals = 0, 0, {}
    for lbl in src_lbl.glob("*.txt"):
        new_text, counts = remap_label(lbl.read_text())
        if not new_text:                   # nothing of interest left -> skip
            skipped += 1
            continue
        img = find_image(lbl.stem, src_img)
        if img is None:
            skipped += 1
            continue
        shutil.copy(img, dst_img / f"{PREFIX}{img.name}")
        (dst_lbl / f"{PREFIX}{lbl.stem}.txt").write_text(new_text)
        added += 1
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
    return {"added": added, "skipped": skipped, "class_counts": totals}


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"{SOURCE} not found. Download it first:\n"
            '  python -c "from ultralytics.data.utils import check_det_dataset; '
            "check_det_dataset('construction-ppe.yaml')\""
        )

    print(f"Merging {SOURCE.name} -> {OURS}\n")
    for src_split, dst_split in SPLIT_MAP.items():
        r = merge_split(src_split, dst_split)
        named = {OUR_NAMES[k]: v for k, v in sorted(r["class_counts"].items())}
        print(f"{src_split:5s} -> {dst_split:5s}: "
              f"+{r['added']} images ({r['skipped']} skipped)  labels: {named}")
    print("\nDone. Retrain to use the enlarged dataset:  python src/train.py")


if __name__ == "__main__":
    main()
