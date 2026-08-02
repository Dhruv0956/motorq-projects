import cv2

# Paths
image_path = r"E:\MH-Weed16An Indian Multiclass Annotated Weed Dataset for Computer Vision Tasks\MH-Weed16\Crop with Weeds\intel Real Sense Depth_Clicks\09082230r1jf472414_145.jpeg"
label_path = r"E:\MH-Weed16An Indian Multiclass Annotated Weed Dataset for Computer Vision Tasks\MH-Weed16\Crop with Weeds\intel Real Sense Depth_Annotations\YOLO_darknet\09082230r1jf472414_145.txt"

# Read image
img = cv2.imread(image_path)
h, w = img.shape[:2]

# Read YOLO labels
with open(label_path, "r") as f:
    lines = f.readlines()

for line in lines:
    cls, x, y, bw, bh = map(float, line.split())

    # Convert normalized coords to pixels
    x1 = int((x - bw / 2) * w)
    y1 = int((y - bh / 2) * h)
    x2 = int((x + bw / 2) * w)
    y2 = int((y + bh / 2) * h)

    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Put class id
    cv2.putText(
        img,
        f"Class {int(cls)}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

# Fit image to screen
screen_w, screen_h = 1920, 1080  # adjust if needed
scale = min(screen_w / w, screen_h / h, 1.0)

if scale < 1.0:
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

# Show image
cv2.imshow("YOLO Boxes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()