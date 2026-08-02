import argparse
import ctypes
import os
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def current_rss_mb() -> float:
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)

        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int

        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.WorkingSetSize / (1024 * 1024)

    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024 if rss > 10_000 else rss / (1024 * 1024)


def load_images(source: str, limit: int, imgsz: int) -> list[np.ndarray]:
    if source == "synthetic":
        return [np.zeros((imgsz, imgsz, 3), dtype=np.uint8) for _ in range(limit)]

    path = Path(source)
    if path.is_dir():
        frames = []
        for image_path in sorted(path.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            frame = cv2.imread(str(image_path))
            if frame is not None:
                frames.append(frame)
            if len(frames) >= limit:
                break
        if frames:
            return frames
        raise RuntimeError(f"No readable images found in {path}")

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    frames = []
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from {source}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local edge detector latency and memory.")
    parser.add_argument("--model", required=True, help="Path to .pt or exported model supported by Ultralytics.")
    parser.add_argument("--source", default="synthetic", help="synthetic, camera index, video path, or image folder.")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    frames = load_images(args.source, args.frames, args.imgsz)
    model = YOLO(str(model_path))

    for frame in frames[: args.warmup]:
        model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

    latencies_ms = []
    rss_start = current_rss_mb()
    start = time.perf_counter()

    for index in range(args.frames):
        frame = frames[index % len(frames)]
        before = time.perf_counter()
        model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        latencies_ms.append((time.perf_counter() - before) * 1000)

    elapsed = time.perf_counter() - start
    rss_end = current_rss_mb()

    print(f"model: {model_path}")
    print(f"source: {args.source}")
    print(f"imgsz: {args.imgsz}")
    print(f"frames: {args.frames}")
    print(f"fps: {args.frames / elapsed:.2f}")
    print(f"latency_avg_ms: {statistics.mean(latencies_ms):.2f}")
    print(f"latency_p95_ms: {statistics.quantiles(latencies_ms, n=20)[18]:.2f}")
    print(f"rss_start_mb: {rss_start:.1f}")
    print(f"rss_end_mb: {rss_end:.1f}")
    print(f"rss_delta_mb: {rss_end - rss_start:.1f}")


if __name__ == "__main__":
    main()
