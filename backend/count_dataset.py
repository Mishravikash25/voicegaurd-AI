"""Quick script to count audio files in the FoR dataset."""
import os
from pathlib import Path

DATASET_ROOT = Path(r"S:\project_train")

variants = {
    "for-2sec": DATASET_ROOT / "for-2sec" / "for-2seconds",
    "for-rerec": DATASET_ROOT / "for-rerec" / "for-rerecorded",
    "for-norm": DATASET_ROOT / "for-norm" / "for-norm",
    "for-original": DATASET_ROOT / "for-original" / "for-original",
}

splits = ["training", "testing", "validation"]
classes = ["fake", "real"]
exts = {'.wav', '.mp3', '.mp4', '.flac', '.ogg'}

for vname, vpath in variants.items():
    print(f"\n{'='*50}")
    print(f"  {vname}  ({vpath})")
    print(f"{'='*50}")
    total = 0
    for split in splits:
        for cls in classes:
            d = vpath / split / cls
            if d.exists():
                count = sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in exts)
            else:
                count = 0
                print(f"  [MISSING] {split}/{cls}")
            print(f"  {split:12s}/{cls:5s}: {count:>7,}")
            total += count
    print(f"  {'TOTAL':>19s}: {total:>7,}")
