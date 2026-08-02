import os
from ultralytics import YOLO
import torch

def main():
    # Path to your final weights (Update the folder number if yours isn't train11)
    model_path = r"C:\Users\Dhruv\Documents\ACE\runs\detect\train13\weights\best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find best.pt at {model_path}. Please verify the folder number.")
        return

    print("Loading Stage 1 Model for Safe Validation...")
    model = YOLO(model_path)

    # Run validation with ultra-safe memory settings
    metrics = model.val(
        data="configs/dataset_stage1.yaml",
        batch=4,          # Low batch size to prevent OpenCV memory allocation errors
        workers=0,        # Zero workers prevents multiprocessing memory leaks
        plots=True,       # Let's generate the precision/recall charts safely now!
        device=0
    )
    
    print("\n=============================================")
    print("STAGE 1 BINARY FIELD DETECTOR METRICS")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print("=============================================")

if __name__ == "__main__":
    main()