from geometry import PoseFeatureExtractor

class PoseFeedbackEngine:

    GREEN_THRESHOLD = 5
    YELLOW_THRESHOLD = 15

    def generate_feedback(self, user_angles, ideal_angles):

        feedback = []

        for i, (user, ideal) in enumerate(zip(user_angles, ideal_angles)):

            difference = user - ideal

            if abs(difference) < self.GREEN_THRESHOLD:
                severity = "green"

            elif abs(difference) < self.YELLOW_THRESHOLD:
                severity = "yellow"

            else:
                severity = "red"

            feedback.append(
                {
                    "connection": PoseFeatureExtractor.CONNECTIONS[i],
                    "user": user,
                    "ideal": ideal,
                    "difference": difference,
                    "severity": severity,
                }
            )

        return feedback