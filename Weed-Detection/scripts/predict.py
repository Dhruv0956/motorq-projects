from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/runs/detect/weed_crop2/weights/best.pt")

# Run prediction
results = model.predict(
    source="C:\\Users\\Dhruv\\Desktop\\MH-Weed16\\Crop with Weeds\\val\\images",   # image path
    save=True,           # save output image
    save_txt=True,       # save output labels
    conf=0.25            # confidence threshold
)

print("Prediction complete")

#9 external pretrained
#10 field weeds single class
#11 weed crop 2
#12 weed crop 5 - v shit
#13 train 8 - sus
#14 train 9 - nothing detected
#15 train 13 - only the smallest one