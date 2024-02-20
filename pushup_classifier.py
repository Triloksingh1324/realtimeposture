import mediapipe as mp
import cv2
from angle_calculator import AngleCalculator

class PushupClassifier:
    def __init__(self):
        self.prev_state = None
        self.pushup_count = 0

    def classify(self, landmarks, prev_state, output_image):
        label = ''
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

            if left_elbow_angle > 150 and left_elbow_angle < 190 or right_elbow_angle > 150 and right_elbow_angle < 190:
                if left_knee_angle > 180 or right_knee_angle > 180:
                    if prev_state != 'down':
                        label = 'down'
                        prev_state = 'down'
            elif left_elbow_angle > 40 and left_elbow_angle < 90 or right_elbow_angle > 40 and right_elbow_angle < 90:
                if left_knee_angle > 180 or right_knee_angle > 180:
                    if prev_state == 'down':
                        self.pushup_count += 1
                        prev_state = 'up'
                        label = 'up'

        cv2.putText(output_image, f'Pushups: {self.pushup_count}', (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        return output_image, label, prev_state
