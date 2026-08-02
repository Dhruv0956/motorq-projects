import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


DEFAULT_DATASETS = {
    "combined": "E:/cropandweed-dataset-main/stage1_final_plant_yolo/data.yaml",
    "cnw": "E:/cropandweed-dataset-main/stage1_final_plant_yolo/data_cnw_val.yaml",
    "mh16": "E:/cropandweed-dataset-main/stage1_final_plant_yolo/data_mh16_val.yaml",
}


def f1(precision, recall):
    total = precision + recall
    return 0.0 if total == 0 else 2.0 * precision * recall / total


def metric_box(results, attr):
    value = getattr(results.box, attr)
    return float(value() if callable(value) else value)


def run_sweep(args):
    model = YOLO(args.model, task="detect")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "imgsz",
        "conf",
        "iou",
        "precision",
        "recall",
        "f1",
        "map50",
        "map50_95",
    ]
    rows = []
    completed = set()
    if output.exists() and args.resume:
        with output.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for key in ("imgsz", "conf", "iou", "precision", "recall", "f1", "map50", "map50_95"):
                    row[key] = float(row[key])
                rows.append(row)
                completed.add((row["dataset"], float(row["conf"]), float(row["iou"])))

    write_header = not output.exists() or not args.resume
    csv_handle = output.open("a" if args.resume else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    for dataset_name, data_yaml in DEFAULT_DATASETS.items():
        if args.datasets and dataset_name not in args.datasets:
            continue

        for conf in args.conf:
            for iou in args.iou:
                key = (dataset_name, conf, iou)
                if key in completed:
                    print(f"Skipping existing dataset={dataset_name} conf={conf} iou={iou}")
                    continue

                print(f"Validating dataset={dataset_name} imgsz={args.imgsz} conf={conf} iou={iou}")
                results = model.val(
                    data=data_yaml,
                    imgsz=args.imgsz,
                    conf=conf,
                    iou=iou,
                    device=args.device,
                    plots=False,
                    verbose=False,
                    batch=args.batch,
                    workers=args.workers,
                )

                precision = metric_box(results, "mp")
                recall = metric_box(results, "mr")
                map50 = metric_box(results, "map50")
                map5095 = metric_box(results, "map")
                row = {
                    "dataset": dataset_name,
                    "imgsz": args.imgsz,
                    "conf": conf,
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1(precision, recall),
                    "map50": map50,
                    "map50_95": map5095,
                }
                rows.append(row)
                writer.writerow(row)
                csv_handle.flush()

                print(
                    "  "
                    f"P={precision:.4f} R={recall:.4f} F1={row['f1']:.4f} "
                    f"mAP50={map50:.4f} mAP50-95={map5095:.4f}"
                )

    csv_handle.close()

    print(f"\nSaved sweep results to {output}")

    for dataset_name in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset_name]
        best_f1 = max(subset, key=lambda row: row["f1"])
        best_map50 = max(subset, key=lambda row: row["map50"])
        print(
            f"\nBest {dataset_name} by F1: "
            f"conf={best_f1['conf']} iou={best_f1['iou']} "
            f"P={best_f1['precision']:.4f} R={best_f1['recall']:.4f} "
            f"F1={best_f1['f1']:.4f} mAP50={best_f1['map50']:.4f}"
        )
        print(
            f"Best {dataset_name} by mAP50: "
            f"conf={best_map50['conf']} iou={best_map50['iou']} "
            f"P={best_map50['precision']:.4f} R={best_map50['recall']:.4f} "
            f"F1={best_map50['f1']:.4f} mAP50={best_map50['map50']:.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="E:/cropandweed-dataset-main/runs/detect/Stage1_Final_Plant_yolov8n_640/weights/best.pt",
    )
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--conf", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.35, 0.50])
    parser.add_argument("--iou", nargs="+", type=float, default=[0.45, 0.55, 0.65])
    parser.add_argument("--datasets", nargs="+", choices=sorted(DEFAULT_DATASETS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        default="C:/Users/Dhruv/Documents/ACE/pi5_exports/stage1_threshold_sweep_416.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_sweep(parse_args())
