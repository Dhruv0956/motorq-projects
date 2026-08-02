import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path("C:/Users/Dhruv/Documents/ACE")
BASE_MODEL = Path("E:/cropandweed-dataset-main/runs/detect/Stage1_Final_Plant_yolov8n_640/weights/best.pt")
DATA_YAML = Path("E:/cropandweed-dataset-main/stage1_final_plant_yolo/data.yaml")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This training recipe expects the laptop GPU.")

    torch.cuda.empty_cache()

    model = YOLO(str(BASE_MODEL))
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        patience=5,
        imgsz=416,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(ROOT / "runs" / "detect"),
        name="Stage1_Improve_416_lowlr",
        exist_ok=True,
        single_cls=True,
        val=True,
        plots=False,
        save=True,
        save_period=1,
        cache=False,
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.2,
        weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=0,
        close_mosaic=3,
        amp=False,
        deterministic=False,
        seed=0,
        mosaic=0.1,
        mixup=0.0,
        copy_paste=0.0,
        degrees=5.0,
        translate=0.05,
        scale=0.25,
        shear=0.5,
        perspective=0.0002,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.01,
        hsv_s=0.35,
        hsv_v=0.25,
        erasing=0.0,
    )


if __name__ == "__main__":
    main()
