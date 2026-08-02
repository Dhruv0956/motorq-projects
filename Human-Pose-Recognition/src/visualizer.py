import cv2
from pathlib import Path


class PoseVisualizer:

    COLORS = {
        "green": (0, 255, 0),
        "yellow": (0, 255, 255),
        "red": (0, 0, 255),
    }

    def visualize(
        self,
        image,
        skeleton,
        feedback,
        pose,
        confidence,
        template_image=None,
    ):

        output = image.copy()

        for item in feedback:

            start_name, end_name = item["connection"]

            p1 = getattr(skeleton, start_name)
            p2 = getattr(skeleton, end_name)

            color = self.COLORS[item["severity"]]

            cv2.line(
                output,
                (int(p1.x), int(p1.y)),
                (int(p2.x), int(p2.y)),
                color,
                3,
            )

            cv2.circle(
                output,
                (int(p1.x), int(p1.y)),
                5,
                color,
                -1,
            )

            cv2.circle(
                output,
                (int(p2.x), int(p2.y)),
                5,
                color,
                -1,
            )

        cv2.putText(
            output,
            f"Pose: {pose}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            output,
            f"Confidence: {confidence*100:.1f}%",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        if template_image is not None:

            template = cv2.imread(str(template_image))

            if template is not None:

                template = cv2.resize(template, (180, 180))

                h, w = template.shape[:2]

                output[
                    10 : 10 + h,
                    output.shape[1] - w - 10 : output.shape[1] - 10,
                ] = template

        return output