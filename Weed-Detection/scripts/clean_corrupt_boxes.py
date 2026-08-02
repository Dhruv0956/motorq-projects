import os

# Define paths to your label directories
DATASET_DIR = r"C:\Users\Dhruv\Documents\ACE\datasets\Field_Binary_Dataset"
LABEL_DIRS = [
    os.path.join(DATASET_DIR, "train", "labels"),
    os.path.join(DATASET_DIR, "val", "labels")
]

def sanitize_labels():
    fixed_files_count = 0
    removed_boxes_count = 0

    print("Scanning label files for zero or negative dimensions...")
    
    for label_dir in LABEL_DIRS:
        if not os.path.exists(label_dir):
            print(f"Skipping missing directory: {label_dir}")
            continue
            
        for file_name in os.listdir(label_dir):
            if not file_name.endswith('.txt'):
                continue
                
            file_path = os.path.join(label_dir, file_name)
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            valid_lines = []
            file_was_modified = False
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    # YOLO format: class x_center y_center width height
                    try:
                        w = float(parts[3])
                        h = float(parts[4])
                        
                        if w > 0 and h > 0:
                            valid_lines.append(line)
                        else:
                            removed_boxes_count += 1
                            file_was_modified = True
                    except ValueError:
                        # Skip malformed non-numeric lines
                        file_was_modified = True
                else:
                    file_was_modified = True
                    
            if file_was_modified:
                # Rewrite the file with only the valid bounding boxes left
                with open(file_path, 'w') as f:
                    f.writelines(valid_lines)
                fixed_files_count += 1

    print("\n--- Cleaning Complete ---")
    print(f"Successfully cleaned {fixed_files_count} label files.")
    print(f"Removed {removed_boxes_count} corrupt bounding boxes.")

if __name__ == "__main__":
    sanitize_labels()