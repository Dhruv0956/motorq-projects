from pathlib import Path

from ultralytics import YOLO


WEIGHT_CANDIDATES = [
    Path("runs/detect/weed_crop2/weights/best.pt"),
    Path("runs/detect/runs/detect/weed_crop2/weights/best.pt"),
]


def find_weights() -> Path | None:
    for candidate in WEIGHT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    weights = find_weights()
    if weights is None:
        print("Missing fine-tuned weights. Checked:")
        for candidate in WEIGHT_CANDIDATES:
            print(f"- {candidate}")
        return

    model = YOLO(str(weights))
    metrics = model.val(
        data="configs/dataset_stage1.yaml",
        batch=2,
        workers=0,
        imgsz=640,
        max_det=50,
        plots=True,
        project="runs/detect",
        name="weed2_on_mh16_val",
        exist_ok=True,
    )

    print("\n=============================================")
    print("STAGE 1 EXTERNAL PRETRAIN + MH16 FINETUNE METRICS")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print("=============================================")


if __name__ == "__main__":
    main()
