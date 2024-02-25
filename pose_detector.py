import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.4, model_complexity=2)
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_pose(self, image):
        output_image = image.copy()
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imageRGB)

        # Draw pose landmarks on the output image
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(output_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

        return output_image, results.pose_landmarks
