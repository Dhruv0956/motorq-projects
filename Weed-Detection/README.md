# Edge Weed And Plant Detector

Local object-detection pipeline for detecting `weed` and `crop` plants with bounding boxes on low-power edge hardware. The target is at least 1 FPS with the smallest practical model and RAM footprint, with no network call during inference.

## Recommended Approach

- Model family: start with a nano YOLO detector (`yolo11n.pt` or the current nano model supported by your Ultralytics install).
- Classes: `weed`, `crop`.
- Input size: begin at `320x320`; only move to `416` or `512` if small weeds are missed.
- Runtime target:
  - Raspberry Pi / ARM CPU: `ncnn` or `tflite` INT8.
  - Intel CPU/NPU: `openvino` INT8/FP16.
  - NVIDIA Jetson: `engine` / TensorRT FP16 or INT8.
  - Generic PC CPU: `onnx` with ONNX Runtime.
- Inference rate: throttle capture to 1 FPS and run detection locally on each selected frame.

## Dataset Layout

Use YOLO detection labels, one text file per image:

```text
datasets/weeds/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Each label row:

```text
class_id x_center y_center width height
```

Values are normalized from `0` to `1`. For this project:

```text
0 = weed
1 = crop
```

## Install

Use a local virtual environment. These dependencies are for training and export. The final edge device only needs the runtime for the exported model format you choose.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

Before training, validate the label files:

```powershell
python scripts/check_dataset.py --data configs/weed_crop.yaml
```

```powershell
python scripts/train_export.py train --data configs/weed_crop.yaml --model yolo11n.pt --imgsz 320 --epochs 80
```

For a very small dataset, freeze less and use more augmentation by default before trying a larger model. Good field coverage matters more than a bigger network: different soil colors, lighting, growth stages, camera heights, blur, shadows, and occlusion.

## Export For Edge

ONNX with built-in NMS:

```powershell
python scripts/train_export.py export --weights runs/detect/weed_crop/weights/best.pt --format onnx --imgsz 320
```

TFLite INT8, useful for ARM/mobile-class CPUs:

```powershell
python scripts/train_export.py export --weights runs/detect/weed_crop/weights/best.pt --format tflite --imgsz 320 --int8 --data configs/weed_crop.yaml
```

NCNN, often a good Raspberry Pi / ARM choice:

```powershell
python scripts/train_export.py export --weights runs/detect/weed_crop/weights/best.pt --format ncnn --imgsz 320
```

TensorRT for Jetson:

```powershell
python scripts/train_export.py export --weights runs/detect/weed_crop/weights/best.pt --format engine --imgsz 320 --half
```

## Local 1 FPS Inference

For the simplest local validation:

```powershell
python scripts/detect_local.py --model runs/detect/weed_crop/weights/best.pt --source 0 --fps 1 --imgsz 320
```

This uses the model locally and skips frames to keep processing at the requested FPS. For production, use the exported runtime artifact instead of the PyTorch `.pt` model.

## Benchmark On The Edge Device

Run this on the target hardware after exporting the model:

```powershell
python scripts/benchmark_local.py --model runs/detect/weed_crop/weights/best.pt --source synthetic --imgsz 320 --frames 60
```

For a real camera or field video:

```powershell
python scripts/benchmark_local.py --model runs/detect/weed_crop/weights/best.pt --source 0 --imgsz 320 --frames 60
```

Track:

- `fps`: must be at least `1.00`.
- `latency_p95_ms`: should stay below `1000 ms` for a 1 FPS system.
- `rss_end_mb`: working memory used by the local process.
- `rss_delta_mb`: memory growth during repeated inference.

## Edge Optimization Checklist

- Keep only two classes unless more categories are operationally required.
- Use `imgsz=320` first; it sharply reduces memory and compute.
- Export with INT8 quantization when the backend supports it.
- Calibrate INT8 export with real field images, not clean lab samples.
- Keep confidence threshold conservative at first (`0.35` to `0.50`) and tune from validation data.
- Limit max detections per frame if dense false positives become expensive.
- Prefer fixed camera height and angle; it reduces dataset and model burden.
- Benchmark on-device, not on the development laptop.

## Acceptance Target

The first production candidate should meet:

- Runs with network disabled.
- Sustains `>= 1 FPS` on the target edge device.
- Uses exported model runtime, not cloud inference.
- Detects `weed` and `crop` bounding boxes with acceptable field validation mAP/precision/recall.
- RAM stays within the target device budget during a 10-minute continuous run.
