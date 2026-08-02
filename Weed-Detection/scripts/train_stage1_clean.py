import os
from ultralytics import YOLO
import torch

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

def main():
    if not torch.cuda.is_available():
        print("CUDA Error")
        return

    torch.cuda.empty_cache()

    model = YOLO("yolov8n.pt")

    model.train(
        data="configs/dataset_stage1.yaml",
        epochs=100,
        patience=30,
        imgsz=640,
        batch=16,
        workers=2,
        cache=False,
        device=0,

        project="runs/detect",
        name="Stage1_Field_Weed_SingleClass_yolov8n_640",
        exist_ok=False,

        single_cls=True,
        val=True,
        plots=False,

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