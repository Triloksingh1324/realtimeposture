import cv2
from angle_calculator import AngleCalculator
from pushup_classifier import PushupClassifier
from pose_detector import PoseDetector
from bicep_classifier import BicepClassifier

# Initialize PushupClassifier
pushup_classifier = PushupClassifier()

# Initialize PoseDetector
pose_detector = PoseDetector()

# Initialize BicepClassifier
bicep_classifier = BicepClassifier()

# Start capturing video
video = cv2.VideoCapture(0)

# Prompt user for choice
choice = input("Enter '1' for Push-up Counting or '2' for Bicep Classification: ")

while True:
    # Read frame from the camera
    ret, frame = video.read()
    if not ret:
        break

    # Detect pose in the frame
    output_image, pose_landmarks = pose_detector.detect_pose(frame)

    # Check if pose landmarks are detected
    if pose_landmarks:
        # Convert landmarks to a list for easier access
        landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in pose_landmarks.landmark]

        # Classify pushup or bicep based on user choice
        if choice == '1':
            output_image, _, pushup_classifier.prev_state = pushup_classifier.classify_pushup(landmarks, pushup_classifier.prev_state, output_image)
        elif choice == '2':
            output_image, _, bicep_classifier.prev_state = bicep_classifier.classify_bicep_pose(landmarks, bicep_classifier.prev_state, output_image)

    # Display the frame
    cv2.imshow('Frame', output_image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
video.release()
cv2.destroyAllWindows()
