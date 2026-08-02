# Weed/Crop Detector Result Report

Run analyzed: `runs/detect/runs/detect/weed_crop2`

## Dataset

- Train images: `1040`
- Validation images: `260`
- Train boxes: `1703`
- Validation boxes: `369`
- Class mapping:
  - `0`: `crop`
  - `1`: `weed`
- Train class counts:
  - `crop`: `1004`
  - `weed`: `699`
- Validation class counts:
  - `crop`: `208`
  - `weed`: `161`

## Training Result

- Model: `YOLO11n`
- Input size: `320`
- Epochs: `80`
- Best epoch by `mAP50-95`: `73`
- Best aggregate metrics:
  - Precision: `0.86174`
  - Recall: `0.81873`
  - mAP50: `0.89607`
  - mAP50-95: `0.61438`
- Final epoch metrics:
  - Precision: `0.88314`
  - Recall: `0.80687`
  - mAP50: `0.89577`
  - mAP50-95: `0.61062`

The curves are healthy: training and validation losses continue trending down, while mAP plateaus near the end. There is no obvious late overfitting spike.

## PyTorch Validation

Weights: `runs/detect/runs/detect/weed_crop2/weights/best.pt`

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 260 | 369 | 0.861 | 0.821 | 0.896 | 0.613 |
| crop | 131 | 208 | 0.829 | 0.822 | 0.890 | 0.652 |
| weed | 129 | 161 | 0.893 | 0.820 | 0.902 | 0.574 |

Weed detection has strong mAP50, but lower mAP50-95 than crop. That means the model often finds weeds, but weed box tightness is less consistent.

## ONNX Export

Exported model: `runs/detect/runs/detect/weed_crop2/weights/best.onnx`

- PyTorch weight size: `5.2 MB`
- ONNX model size: `10.0 MB`
- ONNX validation:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 260 | 369 | 0.869 | 0.803 | 0.884 | 0.632 |
| crop | 131 | 208 | 0.829 | 0.791 | 0.870 | 0.672 |
| weed | 129 | 161 | 0.910 | 0.815 | 0.898 | 0.592 |

The ONNX export preserved quality well. Recall is slightly lower, but mAP50-95 is slightly higher.

## Local Benchmark

Machine: Windows laptop CPU via Ultralytics wrapper.

PyTorch `.pt` benchmark on synthetic frames:

- FPS: `13.60`
- Average latency: `73.55 ms`
- p95 latency: `87.72 ms`
- Working set: about `551 MB`

ONNX benchmark on synthetic frames:

- FPS: `4.90`
- Average latency: `203.89 ms`
- p95 latency: `285.21 ms`
- Working set: about `579 MB`

Both exceed the `1 FPS` requirement on this machine. The memory footprint here includes Python and the Ultralytics wrapper, so production edge memory should be measured with the final native runtime.

## Recommendation

This is a good first deployable candidate. Next, benchmark on the actual edge device and export to the runtime that matches the hardware:

- Intel edge CPU/NPU: export `openvino`.
- Raspberry Pi / ARM Linux: export `ncnn` or `tflite int8`.
- NVIDIA Jetson: export TensorRT `engine`.
- Generic CPU prototype: use the exported `best.onnx`.

Before retraining, test the model on real field video frames from the intended camera height. If weed boxes are loose or missed, collect more examples of small weeds, partial occlusion, soil texture variation, and harsh lighting.
