from pathlib import Path
import csv
import cv2
import mediapipe as mp

from detector import PoseDetector
from skeleton import LandmarkConverter
from geometry import PoseFeatureExtractor


class DatasetBuilder:
    def __init__(self):
        self.detector = PoseDetector()
        self.converter = LandmarkConverter()
        self.extractor = PoseFeatureExtractor()

    def build(
        self,
        dataset_path,
        output_csv,
        image_extensions=(".jpg", ".jpeg", ".png"),
    ):
        dataset_path = Path(dataset_path)
        output_csv = Path(output_csv)

        with open(output_csv, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)

            # Create header automatically
            header = [
                f"angle_{i+1}"
                for i in range(len(PoseFeatureExtractor.CONNECTIONS))
            ]
            header.append("label")
            writer.writerow(header)

            for class_folder in dataset_path.iterdir():
                if not class_folder.is_dir():
                    continue

                label = class_folder.name
                print(f"Processing {label}...")

                for image_path in class_folder.iterdir():

                    if image_path.suffix.lower() not in image_extensions:
                        continue

                    # Read image
                    try:
                        image = self._read_image(image_path)
                    except ValueError as exc:
                        print(f"Skipping {image_path.name} ({exc})")
                        continue

                    # Detect pose
                    try:
                        result = self.detector.detect(image)
                    except ValueError as exc:
                        print(f"Skipping {image_path.name} ({exc})")
                        continue

                    # Convert to Skeleton
                    skeleton = self.converter.convert(result, image)

                    if skeleton is None:
                        print(f"Skipping {image_path.name} (no valid pose)")
                        continue

                    # Extract feature vector
                    angles = self.extractor.extract(skeleton)

                    # Write one sample
                    writer.writerow(angles + [label])

    def _read_image(self, image_path):
        data = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if data is None:
            raise ValueError("image could not be read")

        if data.ndim == 2:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
            return mp.Image(image_format=mp.ImageFormat.SRGB, data=data)

        channels = data.shape[2]
        if channels == 1:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
            return mp.Image(image_format=mp.ImageFormat.SRGB, data=data)
        if channels == 3:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            return mp.Image(image_format=mp.ImageFormat.SRGB, data=data)
        if channels == 4:
            data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA)
            return mp.Image(image_format=mp.ImageFormat.SRGBA, data=data)

        raise ValueError(f"unsupported image with {channels} channels")
