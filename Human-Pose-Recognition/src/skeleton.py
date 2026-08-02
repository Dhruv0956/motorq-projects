from dataclasses import dataclass
from mediapipe.tasks.python.vision import PoseLandmark

@dataclass
class Point:
    x: float
    y: float

@dataclass
class Skeleton:
    neck: Point

    left_shoulder: Point
    right_shoulder: Point

    left_elbow: Point
    right_elbow: Point

    left_wrist: Point
    right_wrist: Point

    left_hip: Point
    right_hip: Point

    left_knee: Point
    right_knee: Point

    left_ankle: Point
    right_ankle: Point

class LandmarkConverter:
    LANDMARK_MAP = {
        "left_shoulder": PoseLandmark.LEFT_SHOULDER,
        "right_shoulder": PoseLandmark.RIGHT_SHOULDER,
        "left_elbow": PoseLandmark.LEFT_ELBOW,
        "right_elbow": PoseLandmark.RIGHT_ELBOW,
        "left_wrist": PoseLandmark.LEFT_WRIST,
        "right_wrist": PoseLandmark.RIGHT_WRIST,
        "left_hip": PoseLandmark.LEFT_HIP,
        "right_hip": PoseLandmark.RIGHT_HIP,
        "left_knee": PoseLandmark.LEFT_KNEE,
        "right_knee": PoseLandmark.RIGHT_KNEE,
        "left_ankle": PoseLandmark.LEFT_ANKLE,
        "right_ankle": PoseLandmark.RIGHT_ANKLE
    }

    def _normalized_to_point(self, landmark, image) -> Point:
        x = landmark.x * image.width
        y = landmark.y * image.height
        return Point(x, y)
    
    def _compute_neck(self, left_shoulder: Point, right_shoulder: Point) -> Point:
        x = (left_shoulder.x + right_shoulder.x) / 2
        y = (left_shoulder.y + right_shoulder.y) / 2
        return Point(x, y)

    def convert(self, result, image):
        if not result.pose_landmarks:
            return None
        landmarks = result.pose_landmarks[0]
        points = {}
        for name, landmark_enum in self.LANDMARK_MAP.items():
            landmark = landmarks[landmark_enum.value]
            points[name] = self._normalized_to_point(landmark, image)
        points["neck"] = self._compute_neck(points["left_shoulder"], points["right_shoulder"])
        return Skeleton(**points)
    