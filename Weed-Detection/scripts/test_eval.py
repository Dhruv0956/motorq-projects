from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/runs/detect/weed_crop5/weights/best.pt")
    metrics = model.val(data="configs/data.yaml", split="test")
    print(metrics)

if __name__ == "__main__":
    main()