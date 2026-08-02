import argparse
import os
import statistics
import time
from pathlib import Path

import cv2
import psutil
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_images(source):
    path = Path(source)
    if path.is_file():
        yield str(path)
        return

    for item in sorted(path.iterdir()):
        if item.suffix.lower() in IMAGE_EXTS:
            yield str(item)


def summarize(values):
    if not values:
        return {"avg": 0.0, "median": 0.0, "p95": 0.0}
    values = sorted(values)
    p95_idx = min(len(values) - 1, int(len(values) * 0.95))
    return {
        "avg": statistics.mean(values),
        "median": statistics.median(values),
        "p95": values[p95_idx],
    }


def run_images(model, source, imgsz, conf, max_images, warmup):
    paths = list(iter_images(source))
    if max_images:
        paths = paths[:max_images]

    if not paths:
        raise FileNotFoundError(f"No images found in {source}")

    for path in paths[:warmup]:
        model.predict(path, imgsz=imgsz, conf=conf, verbose=False)

    timings = []
    box_counts = []
    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss

    for path in paths:
        start = time.perf_counter()
        result = model.predict(path, imgsz=imgsz, conf=conf, verbose=False)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings.append(elapsed_ms)
        box_counts.append(len(result.boxes))
        peak_rss = max(peak_rss, process.memory_info().rss)

    stats = summarize(timings)
    fps = 1000.0 / stats["avg"] if stats["avg"] else 0.0

    return {
        "frames": len(paths),
        "avg_ms": stats["avg"],
        "median_ms": stats["median"],
        "p95_ms": stats["p95"],
        "fps": fps,
        "avg_boxes": statistics.mean(box_counts) if box_counts else 0.0,
        "peak_rss_mb": peak_rss / (1024 * 1024),
    }


def run_camera(model, camera, imgsz, conf, frames, warmup):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera}")

    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss
    timings = []
    box_counts = []
    frame_idx = 0

    try:
        while frame_idx < frames + warmup:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera frame read failed")

            start = time.perf_counter()
            result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if frame_idx >= warmup:
                timings.append(elapsed_ms)
                box_counts.append(len(result.boxes))
                peak_rss = max(peak_rss, process.memory_info().rss)

            frame_idx += 1
    finally:
        cap.release()

    stats = summarize(timings)
    fps = 1000.0 / stats["avg"] if stats["avg"] else 0.0

    return {
        "frames": len(timings),
        "avg_ms": stats["avg"],
        "median_ms": stats["median"],
        "p95_ms": stats["p95"],
        "fps": fps,
        "avg_boxes": statistics.mean(box_counts) if box_counts else 0.0,
        "peak_rss_mb": peak_rss / (1024 * 1024),
    }


def print_report(args, result):
    print("\nRaspberry Pi 5 NCNN Benchmark")
    print(f"model: {args.model}")
    print(f"imgsz: {args.imgsz}")
    print(f"conf: {args.conf}")
    print(f"mode: {args.mode}")
    print(f"frames: {result['frames']}")
    print(f"avg_ms: {result['avg_ms']:.2f}")
    print(f"median_ms: {result['median_ms']:.2f}")
    print(f"p95_ms: {result['p95_ms']:.2f}")
    print(f"fps: {result['fps']:.2f}")
    print(f"avg_boxes: {result['avg_boxes']:.2f}")
    print(f"peak_rss_mb: {result['peak_rss_mb']:.1f}")

    if result["fps"] >= 6.0 and result["peak_rss_mb"] < 5000:
        print("status: PASS for 5-6 FPS and <5GB RAM target")
    elif result["fps"] >= 5.0 and result["peak_rss_mb"] < 5000:
        print("status: BORDERLINE PASS for minimum 5 FPS target")
    else:
        print("status: FAIL target, try smaller imgsz/model or optimize runtime")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to NCNN model folder")
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--mode", choices=["images", "camera"], default="images")
    parser.add_argument("--source", help="Image file or folder for image benchmark")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=200)
    args = parser.parse_args()

    model = YOLO(args.model, task="detect")

    if args.mode == "images":
        if not args.source:
            raise ValueError("--source is required when --mode images")
        result = run_images(model, args.source, args.imgsz, args.conf, args.max_images, args.warmup)
    else:
        result = run_camera(model, args.camera, args.imgsz, args.conf, args.frames, args.warmup)

    print_report(args, result)


if __name__ == "__main__":
    main()
