import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def open_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def draw_boxes(frame, result, class_names) -> None:
    if result.boxes is None:
        return

    for box in result.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        label = f"{class_names.get(cls_id, cls_id)} {conf:.2f}"
        color = (40, 180, 40) if cls_id == 1 else (40, 40, 220)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local 1 FPS weed/crop detection.")
    parser.add_argument("--model", required=True, help="Path to .pt or exported model supported by Ultralytics.")
    parser.add_argument("--source", default="0", help="Camera index, video path, image path, or stream URL.")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", default=None, help="Optional output video path.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    names = model.names if isinstance(model.names, dict) else {i: name for i, name in enumerate(model.names)}

    cap = open_source(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    writer = None
    frame_interval = 1.0 / max(args.fps, 0.1)
    last_inference = 0.0
    last_result = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.monotonic()
        if now - last_inference >= frame_interval:
            last_result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            last_inference = now

        if last_result is not None:
            draw_boxes(frame, last_result, names)

        if args.save and writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))

        if writer is not None:
            writer.write(frame)

        if args.show:
            cv2.imshow("weed-crop detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

