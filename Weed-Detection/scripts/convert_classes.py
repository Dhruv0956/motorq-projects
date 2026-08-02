from pathlib import Path

src = Path("C:\\Users\\Dhruv\\Desktop\\MH-Weed16\\Crop with Weeds\\intel Real Sense Depth_CropandWeed_Annotations\\YOLO_darknet")
dst = Path("C:\\Users\\Dhruv\\Desktop\\MH-Weed16\\Crop with Weeds\\intel Real Sense Depth_SingleClass_Annotations")

dst.mkdir(parents=True, exist_ok=True)

for file in src.glob("*.txt"):
    output = []

    for line in file.read_text().splitlines():
        parts = line.split()

        if len(parts) >= 5:
            parts[0] = "1"   # all become weed

        output.append(" ".join(parts))

    (dst / file.name).write_text("\n".join(output))