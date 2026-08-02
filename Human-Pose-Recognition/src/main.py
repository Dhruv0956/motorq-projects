import joblib
import cv2
from pathlib import Path

from predict import PosePredictor
from feedback import PoseFeedbackEngine
from visualizer import PoseVisualizer


predictor = PosePredictor()

feedback_engine = PoseFeedbackEngine()

visualizer = PoseVisualizer()

templates = joblib.load("models/pose_templates.joblib")

image_path = "test.jpg"

result = predictor.predict(image_path)

ideal_angles = templates[result["pose"]]

feedback = feedback_engine.generate_feedback(
    result["angles"],
    ideal_angles,
)

image = cv2.imread(image_path)

template_image = Path("templates") / f"{result['pose']}.jpg"

output = visualizer.visualize(
    image=image,
    skeleton=result["skeleton"],
    feedback=feedback,
    pose=result["pose"],
    confidence=result["confidence"],
    template_image=template_image,
)

cv2.imshow("Pose Feedback", output)
cv2.waitKey(0)
cv2.destroyAllWindows()