from dataclasses import dataclass
import math

@dataclass
class Vector:
    x: float
    y: float

def vector(p1, p2):
    dx = p2.x - p1.x
    dy = - p2.y + p1.y
    return Vector(dx, dy)

def length(v):
    l = math.hypot(v.x, v.y)
    return l

def angle(v):
    ang = math.atan2(v.y, v.x)
    return math.degrees(ang)

class PoseFeatureExtractor:
    CONNECTIONS = [
        ("neck", "right_shoulder"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("neck", "left_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("neck", "right_hip"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("neck", "left_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle")
    ]

    def extract(self, skeleton):
        angles = []
        for connection in self.CONNECTIONS:
            s, t = connection
            a = getattr(skeleton, s)
            b = getattr(skeleton, t)
            v = vector(a, b)
            ang = angle(v)
            angles.append(ang)
        return angles