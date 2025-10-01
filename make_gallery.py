# Copy a small random sample from your dataset's test split into sample_images/
# Run: python make_gallery.py --data_root /path/to/chest_xray --per_class 8
import argparse, random, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Folder with train/ and test/ subfolders")
    ap.add_argument("--per_class", type=int, default=8)
    ap.add_argument("--out_dir", default="sample_images")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = ["NORMAL", "PNEUMONIA"]
    random.seed(42)

    for cls in classes:
        src = data_root / "test" / cls
        imgs = [p for p in src.glob("*.jpg")] + [p for p in src.glob("*.png")] + [p for p in src.glob("*.jpeg")]
        if not imgs:
            print(f"Warning: no images found for class {cls} under {src}")
            continue
        picks = random.sample(imgs, min(args.per_class, len(imgs)))
        dest = out_dir / cls
        dest.mkdir(parents=True, exist_ok=True)
        for p in picks:
            shutil.copy2(p, dest / p.name)

    (out_dir / "README.txt").write_text(
        "Images sampled from the public chest_xray test split.\n"
        "For research/demo use only. Not for clinical use.\n"
    )
    print(f"Done. Copied images to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
