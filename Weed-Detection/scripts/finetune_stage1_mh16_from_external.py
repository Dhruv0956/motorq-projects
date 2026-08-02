import os
from pathlib import Path

import torch
from ultralytics import YOLO


os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


PRETRAINED_WEIGHT_CANDIDATES = [
    Path("runs/detect/Stage1_External_Pretrain_yolov8n_640/weights/best.pt"),
    Path("runs/detect/runs/detect/Stage1_External_Pretrain_yolov8n_640/weights/best.pt"),
]


def find_pretrained_weights() -> Path | None:
    for candidate in PRETRAINED_WEIGHT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA Error")
        return

    pretrained_weights = find_pretrained_weights()
    if pretrained_weights is None:
        print("Missing pretrained weights. Checked:")
        for candidate in PRETRAINED_WEIGHT_CANDIDATES:
            print(f"- {candidate}")
        print("Run scripts/train_stage1_external_pretrain.py first.")
        return

    torch.cuda.empty_cache()

    model = YOLO(str(pretrained_weights))
    model.train(
    data="configs/dataset_stage1.yaml",

    epochs=30,
    patience=10,

    imgsz=640,
    batch=2,

    workers=0,

    cache=False,

    device=0,

    project="runs/detect",
    name="Stage1_MH16_Finetune_FromExternal_yolov8n_640",

    single_cls=True,

    val=True,
    plots=False,

    lr0=0.0005,

    # disable heavy augmentation first
    mosaic=0.0,
    close_mosaic=0,

    degrees=5.0,
    translate=0.05,
    scale=0.3,

    hsv_h=0.01,
    hsv_s=0.5,
    hsv_v=0.3,

    fliplr=0.5,
)


if __name__ == "__main__":
    main()
