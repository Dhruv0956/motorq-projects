import os
from collections import Counter

# Path to the actual pre-made university labels
LABEL_DIR = r"E:\External_CropWeed_Binary\val\labels"

def check_labels():
    if not os.path.exists(LABEL_DIR):
        print(f"Error: Could not find directory at {LABEL_DIR}")
        return

    all_found_classes = []
    txt_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.txt')]
    
    print(f"Scanning {len(txt_files)} pre-made label files...")
    
    for txt_file in txt_files:
        with open(os.path.join(LABEL_DIR, txt_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    all_found_classes.append(int(parts[0]))
                    
    counts = Counter(all_found_classes)
    print("\n--- Found Class IDs and their Instance Counts ---")
    for class_id, count in sorted(counts.items()):
        print(f"Class ID [{class_id}]: {count} bounding boxes")

if __name__ == "__main__":
    check_labels()