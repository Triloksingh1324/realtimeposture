import mediapipe as mp
import cv2
from angle_calculator import AngleCalculator


class TreeClassifier:
    def __init__(self):
        self.prev_state = None

    def classify(self, landmarks, prev_state, output_image):
        label = 'Unknown Pose'
        if landmarks:
            left_knee_angle = AngleCalculator().calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value],
                                                                landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value],
                                                                landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value])
            right_knee_angle = AngleCalculator().calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value],
                                                                 landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value],
                                                                 landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value])
        

             # Check if one leg is straight
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

                # Check if the other leg is bended at the required angle.
                if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:

                    # Specify the label of the pose that is tree pose.
                    label = 'Tree Pose'

        cv2.putText(output_image, label, (10, 60),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        return output_image, label, prev_state
