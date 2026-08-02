import os
from ultralytics import YOLO

def main():
    # Path to the partially trained weights from your crashed run
    # Ultralytics auto-creates this folder structure
    last_weights = r"C:\\Users\\Dhruv\\Documents\\ACE\\runs\\detect\\train8\\weights\\last.pt"
    
    if os.path.exists(last_weights):
        print(f"Found checkpoint! Resuming from: {last_weights}")
        model = YOLO(last_weights)
        
        # Resume training
        model.train(
            resume=True,
            workers=0 # CRITICAL: Sets workers to 0 to bypass the Windows Multiprocessing/DLL bug entirely for the final 10 epochs
        )
    else:
        print("Could not find 'last.pt' at the expected location. Double check the run path.")

if __name__ == "__main__":
    main()