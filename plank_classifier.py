import mediapipe as mp
import cv2
from angle_calculator import AngleCalculator


class PlankClassifier:
    def __init__(self):
        self.prev_state = None

    def classify(self, landmarks, prev_state, output_image):
        label = 'Unknown Pose'
        if landmarks:
            left_elbow_angle = AngleCalculator().calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value],
                                                                 landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value],
                                                                 landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value])
            right_elbow_angle = AngleCalculator().calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                                  landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value],
                                                                  landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value])
            left_knee_angle = AngleCalculator().calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value],
                                                                landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value],
                                                                landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value])
            right_knee_angle = AngleCalculator().calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value],
                                                                 landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value],
                                                                 landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value])
            # Get the angle between the left shoulder, hip, and ankle points.
            left_shoulder_hip_ankle_angle = AngleCalculator().calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value],
                                                                             landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value],
                                                                             landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value])

            # Get the angle between the right shoulder, hip, and ankle points.
            right_shoulder_hip_ankle_angle = AngleCalculator().calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                                              landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value],
                                                                              landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value])

            # Check if both arms are straight and elbows are close to 90 degrees.
            if left_elbow_angle > 80 and left_elbow_angle < 95 and right_elbow_angle > 80 and right_elbow_angle < 95:

                # Check if shoulders, hips, and ankles are roughly aligned (close to 180 degrees).
                if left_shoulder_hip_ankle_angle > 170 and left_shoulder_hip_ankle_angle < 195 and right_shoulder_hip_ankle_angle > 170 and right_shoulder_hip_ankle_angle < 195:

                    # Check if hips are close to the ground (angle between hips and knees is small).
                    if left_knee_angle > 180 and right_knee_angle > 180:

                        label = 'Plank Pose'

        cv2.putText(output_image, label, (10, 60),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        return output_image, label, prev_state
