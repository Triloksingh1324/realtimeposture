import mediapipe as mp
import cv2
from angle_calculator import AngleCalculator


class TposeClassifier:
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
            left_elbow_angle = AngleCalculator().calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value],
                                                                 landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value],
                                                                 landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value])
            right_elbow_angle = AngleCalculator().calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                                  landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value],
                                                                  landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value])
            # Get the angle between the left elbow, shoulder and hip points.
            left_shoulder_angle = AngleCalculator().calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value],
                                                landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value],
                                                landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value])

            # Get the angle between the right hip, shoulder and elbow points.
            right_shoulder_angle = AngleCalculator().calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value],
                                                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value])
            # Check if the both arms are straight.
            if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:

                # Check if shoulders are at the required angle.
                if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:
                    # Check if it is the T pose.
                    #----------------------------------------------------------------------------------------------------------------

                    # Check if both legs are straight
                    if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:

                        # Specify the label of the pose that is tree pose.
                        label = 'T Pose'


        cv2.putText(output_image, label, (10, 60),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        return output_image, label, prev_state
