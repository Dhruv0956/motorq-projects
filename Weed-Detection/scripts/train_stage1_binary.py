import os
from ultralytics import YOLO
import torch

# --- MEMORY STABILITY TUNING FOR WINDOWS ---
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64" # Tighter blocks for laptop GPUs

def main():
    if not torch.cuda.is_available():
        print("CUDA Error: Check GPU availability.")
        return

    # Clear out any fragmented junk left over from the crash
    torch.cuda.empty_cache()

    # Load fresh Nano backbone
    model = YOLO("yolov8n.pt")

    model.train(
        data="configs/dataset_stage1.yaml", 
        epochs=50,
        imgsz=640,
        
        # --- THE STABILITY & SPEED BALANCE ---
        batch=16,            # Drop to 16 to drastically lower active RAM usage
        workers=2,           # Keep it at 2: parallel processing without overwhelming Windows spawn limits
        cache=False,         # CRITICAL: Turning off RAM caching prevents the MemoryError entirely
        
        device=0,
        save=True,
        single_cls=True,     # Collapses all 16 labels into 1 single 'Weed' box
        
        # Streamlined augmentations to lighten CPU math load
        mosaic=0.5,          
        degrees=10.0,    
        hsv_h=0.015,     
        hsv_v=0.4
    )

if __name__ == "__main__":
    main()