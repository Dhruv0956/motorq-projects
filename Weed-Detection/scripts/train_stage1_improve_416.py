import os
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path("C:/Users/Dhruv/Documents/ACE")
BASE_MODEL = Path("E:/cropandweed-dataset-main/runs/detect/Stage1_Final_Plant_yolov8n_640/weights/best.pt")
DATA_YAML = Path("E:/cropandweed-dataset-main/stage1_final_plant_yolo/data.yaml")


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This training recipe expects the laptop GPU.")

    torch.cuda.empty_cache()

    model = YOLO(str(BASE_MODEL))
    model.train(
        data=str(DATA_YAML),
        epochs=30,
        patience=10,
        imgsz=416,
        batch=4,
        workers=2,
        device=0,
        project=str(ROOT / "runs" / "detect"),
        name="Stage1_Improve_416_from_best_b4w2",
        exist_ok=True,
        single_cls=True,
        val=True,
        plots=False,
        save=True,
        save_period=5,
        cache=False,
        optimizer="AdamW",
        lr0=0.0008,
        lrf=0.02,
        weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=3,
        close_mosaic=10,
        amp=False,
        deterministic=False,
        seed=0,
        mosaic=0.35,
        mixup=0.0,
        copy_paste=0.0,
        degrees=8.0,
        translate=0.08,
        scale=0.35,
        shear=1.0,
        perspective=0.0003,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.45,
        hsv_v=0.35,
        erasing=0.0,
    )


if __name__ == "__main__":
    main()
