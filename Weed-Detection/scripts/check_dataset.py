import argparse
from pathlib import Path

import yaml


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_split(root: Path, split_value: str) -> Path:
    split_path = Path(split_value)
    return split_path if split_path.is_absolute() else root / split_path


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        images_index = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"Image path does not include an images folder: {image_path}") from exc
    parts[images_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def validate_label(label_path: Path, class_count: int) -> list[str]:
    errors = []
    if not label_path.exists():
        return [f"missing label: {label_path}"]

    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            values = stripped.split()
            if len(values) != 5:
                errors.append(f"{label_path}:{line_number} expected 5 values, got {len(values)}")
                continue
            try:
                class_id = int(values[0])
                box = [float(value) for value in values[1:]]
            except ValueError:
                errors.append(f"{label_path}:{line_number} contains non-numeric values")
                continue
            if class_id < 0 or class_id >= class_count:
                errors.append(f"{label_path}:{line_number} invalid class id {class_id}")
            if any(value < 0.0 or value > 1.0 for value in box):
                errors.append(f"{label_path}:{line_number} box values must be normalized 0..1")
            if box[2] <= 0.0 or box[3] <= 0.0:
                errors.append(f"{label_path}:{line_number} width and height must be positive")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO weed/crop dataset labels.")
    parser.add_argument("--data", default="configs/weed_crop.yaml")
    args = parser.parse_args()

    config_path = Path(args.data)
    config = read_yaml(config_path)
    root = Path(config["path"])
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()

    names = config["names"]
    class_count = len(names)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_errors = []

    for split in ("train", "val"):
        split_dir = resolve_split(root, config[split])
        images = [path for path in split_dir.rglob("*") if path.suffix.lower() in image_extensions]
        if not images:
            all_errors.append(f"{split}: no images found in {split_dir}")
            continue
        for image_path in images:
            all_errors.extend(validate_label(label_path_for(image_path), class_count))
        print(f"{split}: {len(images)} images checked")

    if all_errors:
        print("\nDataset issues:")
        for error in all_errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Dataset labels look valid.")


if __name__ == "__main__":
    main()

