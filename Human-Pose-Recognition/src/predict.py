import mediapipe as mp
import cv2
from pathlib import Path
from detector import PoseDetector
from geometry import PoseFeatureExtractor
from skeleton import LandmarkConverter
from classifier import PoseClassifier

class PosePredictor:
    def __init__(self, model_path=None):
        default_path = Path(__file__).parent.parent / "models" / "pose_classifier.joblib"
        self.detector = PoseDetector()
        self.converter = LandmarkConverter()
        self.extractor = PoseFeatureExtractor()
        if model_path is None:
            model_path = default_path
        self.classifier = PoseClassifier.load(model_path)

    def predict(self, image_path):
        image = mp.Image.create_from_file(str(image_path))
        return self.predict_image(image)

    def predict_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.predict_image(image)

    def predict_image(self, image):
        result = self.detector.detect(image)
        skeleton = self.converter.convert(result, image)
        if skeleton is None:
            return None
        features = self.extractor.extract(skeleton)
        prediction = self.classifier.predict(features)
        confidence = self.classifier.predict_proba(features).max()
        return {
            "pose": prediction[0],
            "confidence": confidence,
            "angles": features,
            "skeleton": skeleton
        }
