import os

import torch
from ultralytics import YOLO


os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA Error")
        return

    torch.cuda.empty_cache()

    model = YOLO("yolov8n.pt")
    model.train(
        data="configs/external_crop_weed_binary.yaml",
        epochs=80,
        patience=20,
        imgsz=640,
        batch=8,
        workers=0,
        cache=False,
        device=0,
        project="runs/detect",
        name="Stage1_External_Pretrain_yolov8n_640",
        exist_ok=False,
        single_cls=True,
        val=True,
        plots=False,
        max_det=50,
        mosaic=0.4,
        close_mosaic=10,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
    )


if __name__ == "__main__":
    main()

