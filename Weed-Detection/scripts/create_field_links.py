import os
import random
from pathlib import Path

def main():
    # --- RAW UNTOUCHED PATHS ---
    RAW_IMG_DIR = Path(r"C:\\Users\\Dhruv\\Desktop\\MH-Weed16\\Crop with Weeds\\intel Real Sense Depth_Clicks")
    RAW_LBL_DIR = Path(r"C:\\Users\\Dhruv\\Desktop\\MH-Weed16\\Crop with Weeds\\intel Real Sense Depth_Annotations\\YOLO_darknet")
    
    # --- TARGET VIRTUAL WORKSPACE ---
    VIRT_ROOT = Path(r"C:\\Users\\Dhruv\\Documents\\ACE\\datasets\\Field_Binary_Dataset")

    # Clear old attempts safely without touching raw assets
    if VIRT_ROOT.exists():
        import shutil
        print("Cleaning up old virtual workspace...")
        shutil.rmtree(VIRT_ROOT)

    # Gather matching files
    valid_names = []
    for img_path in RAW_IMG_DIR.glob("*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            lbl_path = RAW_LBL_DIR / f"{img_path.stem}.txt"
            if lbl_path.exists():
                valid_names.append((img_path, lbl_path))

    print(f"Found {len(valid_names)} valid field image/label pairs.")

    # 80/20 Split
    random.seed(42)
    random.shuffle(valid_names)
    split_idx = int(len(valid_names) * 0.8)
    
    splits = {
        'train': valid_names[:split_idx],
        'val': valid_names[split_idx:]
    }

    # Generate real symlinks
    for split, pairs in splits.items():
        img_dest_dir = VIRT_ROOT / split / 'images'
        lbl_dest_dir = VIRT_ROOT / split / 'labels'
        img_dest_dir.mkdir(parents=True, exist_ok=True)
        lbl_dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"Creating virtual links for {len(pairs)} files in '{split}'...")
        for src_img, src_lbl in pairs:
            # Create a true Windows link for the image
            os.symlink(src_img, img_dest_dir / src_img.name)
            # Create a true Windows link for the label
            os.symlink(src_lbl, lbl_dest_dir / src_lbl.name)

    print(f"\nVirtual dataset linked beautifully at: {VIRT_ROOT}")

if __name__ == "__main__":
    main()