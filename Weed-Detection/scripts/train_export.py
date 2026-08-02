import argparse
from pathlib import Path

from ultralytics import YOLO


def train(args: argparse.Namespace) -> None:
    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project="runs/detect",
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
    )


def export(args: argparse.Namespace) -> None:
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    model.export(
        format=args.format,
        imgsz=args.imgsz,
        int8=args.int8,
        half=args.half,
        data=args.data,
        nms=args.nms,
        simplify=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and export an edge weed/crop detector.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", default="configs/weed_crop.yaml")
    train_parser.add_argument("--model", default="yolo11n.pt")
    train_parser.add_argument("--imgsz", type=int, default=320)
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--batch", type=int, default=-1)
    train_parser.add_argument("--name", default="weed_crop")
    train_parser.add_argument("--patience", type=int, default=20)
    train_parser.add_argument("--workers", type=int, default=4)
    train_parser.add_argument("--device", default=None)
    train_parser.set_defaults(func=train)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--weights", required=True)
    export_parser.add_argument("--format", default="onnx")
    export_parser.add_argument("--imgsz", type=int, default=320)
    export_parser.add_argument("--data", default="configs/weed_crop.yaml")
    export_parser.add_argument("--int8", action="store_true")
    export_parser.add_argument("--half", action="store_true")
    export_parser.add_argument("--nms", action="store_true", default=True)
    export_parser.set_defaults(func=export)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

