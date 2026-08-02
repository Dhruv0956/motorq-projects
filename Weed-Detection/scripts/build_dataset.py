import os
import shutil
import random

# --- UPDATE THESE PATHS TO MATCH YOUR ACTUAL DIRECTORIES ---
SOURCE_BASE_DIR = "C:\\Users\\Dhruv\\Desktop\\MH-Weed16\\Individual Weed Species\\16 Classes of Weed_Species\\Individual Weed_Species"
OUTPUT_DIR = "datasets/Top5_Weeds_Dataset"

# Define the exact folder names from your layout and map them to Class IDs
TARGET_WEEDS = {
    "1..Lavhala_(Cyperus_Rotundus)": 0,
    "7.Bilayat_(Mexicana_Argemone)": 1,
    "10.Gajar_gavat_(Parthenium hysterophorus)": 2,
    "13.Harali_(Cynodon_dactylon)": 3,
    "15.Punarnava _(Boerhaavia diffusa)": 4
}

# Create YOLO directory structures
for split in ['train', 'val']:
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)

# Process each folder
for folder_name, class_id in TARGET_WEEDS.items():
    source_folder_path = os.path.join(SOURCE_BASE_DIR, folder_name)
    
    if not os.path.exists(source_folder_path):
        print(f"Warning: Folder not found -> {source_folder_path}")
        continue
        
    # Gather all image files (.jpg, .png, etc.)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    images = [f for f in os.listdir(source_folder_path) if f.lower().endswith(valid_extensions)]
    
    # Shuffle for a fair train/val split
    random.seed(42)
    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    def deploy_split(image_list, split_name):
        for img_name in image_list:
            src_img_path = os.path.join(source_folder_path, img_name)
            
            # Form clean unique destination name to avoid duplicates across classes
            clean_name = f"class_{class_id}_{img_name}"
            base_name, _ = os.path.splitext(clean_name)
            
            dest_img_path = os.path.join(OUTPUT_DIR, split_name, 'images', clean_name)
            dest_lbl_path = os.path.join(OUTPUT_DIR, split_name, 'labels', f"{base_name}.txt")
            
            # 1. Copy Image
            shutil.copy(src_img_path, dest_img_path)
            
            # 2. Write Full-Frame YOLO Bounding Box (Class_ID Center_X Center_Y Width Height)
            with open(dest_lbl_path, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

    deploy_split(train_images, 'train')
    deploy_split(val_images, 'val')
    print(f"Successfully processed {len(images)} images for Class {class_id} ({folder_name})")

print(f"\nDataset generation complete! Files are saved inside '{OUTPUT_DIR}'")