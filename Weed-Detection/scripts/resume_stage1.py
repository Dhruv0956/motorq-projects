import os
from ultralytics import YOLO
import torch

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

def main():
    if not torch.cuda.is_available():
        print("CUDA Error")
        return

    # Clear memory fragments from the crash
    torch.cuda.empty_cache()

    # Point directly to the checkpoint file created right before the crash
    # Note: Check if yours saved to train11 or a different number and update if needed
    latest_checkpoint = r"C:\Users\Dhruv\Documents\ACE\runs\detect\train13\weights\last.pt"

    if os.path.exists(latest_checkpoint):
        print(f"Found checkpoint! Resuming Stage 1 from: {latest_checkpoint}")
        model = YOLO(latest_checkpoint)
        
        # Resume training with memory safety overrides
        model.train(
            resume=True,
            workers=2,
            batch=16,
            cache=False,
            single_cls=True,
            
            # --- THE MEMORY SAFETY OVERRIDES ---
            val=False,   # Skip intermediate validation loops to prevent matrix allocation errors
            plots=False  # Do not generate heavy evaluation charts until the final epoch
        )
    else:
        print(f"Could not find checkpoint at: {latest_checkpoint}")
        print("Please verify the exact folder path inside your 'runs/detect/' directory.")

if __name__ == "__main__":
    main()