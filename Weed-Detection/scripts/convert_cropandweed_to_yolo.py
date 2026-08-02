import argparse
import csv
import os
import random
import shutil
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CropAndWeed bbox CSV annotations into YOLO train/val folders."
    )
    parser.add_argument("--source-root", default=r"C:\Users\Dhruv\Desktop\data")
    parser.add_argument("--bbox-variant", default="CropOrWeed2")
    parser.add_argument("--output-root", default="datasets/External_CropWeed_Binary")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="Use hardlink by default to avoid duplicating large images on the same drive.",
    )
    return parser.parse_args()


def image_map(images_dir: Path) -> dict[str, Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return {path.stem: path for path in images_dir.rglob("*") if path.suffix.lower() in extensions}


def write_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    if mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def convert_box(row: dict[str, str], width: int, height: int) -> str | None:
    left = float(row["left"])
    top = float(row["top"])
    right = float(row["right"])
    bottom = float(row["bottom"])

    left = max(0.0, min(left, width))
    right = max(0.0, min(right, width))
    top = max(0.0, min(top, height))
    bottom = max(0.0, min(bottom, height))

    box_width = right - left
    box_height = bottom - top
    if box_width <= 0 or box_height <= 0:
        return None

    x_center = (left + right) / 2 / width
    y_center = (top + bottom) / 2 / height
    norm_width = box_width / width
    norm_height = box_height / height

    class_id = int(row["label_id"])
    return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"


def convert_annotation(csv_path: Path, image_path: Path, label_path: Path) -> int:
    with Image.open(image_path) as image:
        width, height = image.size

    converted = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(
            handle,
            fieldnames=["left", "top", "right", "bottom", "label_id", "stem_x", "stem_y"],
        )
        for row in reader:
            line = convert_box(row, width, height)
            if line is not None:
                converted.append(line)

    label_path.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
    return len(converted)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    images_dir = source_root / "images"
    bboxes_dir = source_root / "bboxes" / args.bbox_variant
    output_root = Path(args.output_root)

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not bboxes_dir.exists():
        raise FileNotFoundError(f"Missing bbox directory: {bboxes_dir}")

    images = image_map(images_dir)
    csv_files = sorted(bboxes_dir.glob("*.csv"))
    matched = [(csv_path, images[csv_path.stem]) for csv_path in csv_files if csv_path.stem in images]
    missing_images = len(csv_files) - len(matched)

    if not matched:
        raise RuntimeError(
            f"No CropAndWeed images matched {bboxes_dir}. "
            f"Found {len(csv_files)} CSV files and {len(images)} images under {images_dir}."
        )

    random.seed(args.seed)
    random.shuffle(matched)
    val_count = max(1, int(len(matched) * args.val_ratio))
    splits = {
        "val": matched[:val_count],
        "train": matched[val_count:],
    }

    total_boxes = 0
    for split, pairs in splits.items():
        image_out = output_root / split / "images"
        label_out = output_root / split / "labels"
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for csv_path, image_path in pairs:
            dst_image = image_out / image_path.name
            dst_label = label_out / f"{image_path.stem}.txt"
            write_link_or_copy(image_path, dst_image, args.link_mode)
            total_boxes += convert_annotation(csv_path, image_path, dst_label)

        print(f"{split}: {len(pairs)} images")

    print(f"matched_csv_images: {len(matched)}")
    print(f"missing_images_for_csv: {missing_images}")
    print(f"total_boxes: {total_boxes}")
    print(f"output_root: {output_root.resolve()}")


if __name__ == "__main__":
    main()

