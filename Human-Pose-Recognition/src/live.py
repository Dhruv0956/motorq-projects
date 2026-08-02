import argparse
from pathlib import Path

import cv2
import joblib

from feedback import PoseFeedbackEngine
from predict import PosePredictor
from visualizer import PoseVisualizer


def run_live(camera_index=0):
    predictor = PosePredictor()
    feedback_engine = PoseFeedbackEngine()
    visualizer = PoseVisualizer()
    templates = joblib.load("models/pose_templates.joblib")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = predictor.predict_frame(frame)

        if result is None:
            output = frame.copy()
            cv2.putText(
                output,
                "No pose detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        else:
            ideal_angles = templates[result["pose"]]
            feedback = feedback_engine.generate_feedback(
                result["angles"],
                ideal_angles,
            )
            template_image = Path("templates") / f"{result['pose']}.jpg"
            output = visualizer.visualize(
                image=frame,
                skeleton=result["skeleton"],
                feedback=feedback,
                pose=result["pose"],
                confidence=result["confidence"],
                template_image=template_image,
            )

        cv2.imshow("Live Pose Feedback", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    run_live(args.camera)
