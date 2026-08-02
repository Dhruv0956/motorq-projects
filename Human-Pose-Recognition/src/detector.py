from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


class PoseDetector:
    def __init__(self, model_path=None):
        default_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "models",
            "pose_landmarker_full.task"
        )
        if model_path is None:
            model_path = default_path

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options
        )

        self._detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, image):
        return self._detector.detect(image)
