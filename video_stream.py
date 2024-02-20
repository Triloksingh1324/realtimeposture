import cv2
from flask import Response
from angle_calculator import AngleCalculator
from pushup_classifier import PushupClassifier
from pose_detector import PoseDetector
from bicep_classifier import BicepClassifier
from plank_classifier import PlankClassifier
from tree_classifier import TreeClassifier
from tpose_classifier import TposeClassifier
from warrior_classifier import WarriorClassifier

class VideoStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.pushup_classifier = PushupClassifier()
        self.bicep_classifier = BicepClassifier()
        self.pose_detector = PoseDetector()
        self.plank_classifier = PlankClassifier()
        self.tree_classifier = TreeClassifier()
        self.tpose_classifier = TposeClassifier()
        self.warrior_classifier =WarriorClassifier()
        self.classifier = None

    def set_classifier(self, choice):
        if choice == 'pushup':
            self.classifier = self.pushup_classifier
        elif choice == 'bicep':
            self.classifier = self.bicep_classifier
        elif choice == 'plank':
            self.classifier = self.plank_classifier
        elif choice == 'Tree':
            self.classifier = self.tree_classifier
        elif choice == 'TPose':
            self.classifier = self.tpose_classifier
        elif choice == 'WarriorPose':
            self.classifier = self.warrior_classifier

    def stream(self):
        def generate_frames():
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                output_image, pose_landmarks = self.pose_detector.detect_pose(frame)
                if pose_landmarks:
                    landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in pose_landmarks.landmark]

                    if self.classifier:
                        _, _, self.classifier.prev_state = self.classifier.classify(landmarks, self.classifier.prev_state, output_image)

                ret, buffer = cv2.imencode('.jpg', output_image)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
