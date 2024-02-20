import cv2
import math
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True,
                    min_detection_confidence=0.7, model_complexity=2)

# Initialize the counter
bicep_count = 0
state=" "
# Initialize variables to track previous state
prev_state = 'down state'
current_state = None
# Initialize a variable to track whether a cycle has been completed
cycle_completed = False
# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils

def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image,
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''

    # Create a copy of the input image.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = pose.process(imageRGB)

    # Retrieve the height and width of the input image.
    height, width, _ = image.shape

    # Initialize a list to store the detected landmarks.
    landmarks = []

    # Check if any landmarks are detected.
    if results.pose_landmarks:

        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS, connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:

            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))

    # Check if the original input image and the resultant image are specified to be displayed.
    if display:

        # Display the original input image and the resultant image.
        plt.figure(figsize=[22, 22])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    # Otherwise
    else:

        # Return the output image and the found landmarks.
        return output_image, landmarks


def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                         math.atan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360

    # Return the calculated angle.
    return angle

def classifyPose(landmarks, output_image, prev_state, cycle_completed, display=False):
    global state
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    global bicep_count
    # Initialize the label of the pose. It is not known at this stage.
    label = ''
    
    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)

    # Calculate the required angles.
    # ----------------------------------------------------------------------------------------------------------------

    # Get the angle between the left shoulder, elbow and wrist points.
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    # Get the angle between the right shoulder, elbow and wrist points.
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    # ----------------------------------------------------------------------------------------------------------------


# Check if biceps
    # Check if down
    if left_elbow_angle > 140 and left_elbow_angle < 170 or right_elbow_angle > 140 and right_elbow_angle < 170:            
                    #  if left_shoulder_angle>240 and left_shoulder_angle <300 and right_shoulder_angle>240 and right_shoulder_angle <300:
                    label = 'down state'
                    if prev_state == 'up state':
                        cycle_completed = True
                        bicep_count = bicep_count + 1
                        print("bicep count:", bicep_count)
                        cycle_completed = False
                        prev_state = 'down state'
                      
    # Check if up
    if left_elbow_angle > 50 and left_elbow_angle < 70 or right_elbow_angle > 50 and right_elbow_angle < 70:
                    label= 'up state'
                    prev_state = 'up state'
                    
                      
    # else:
    #     if left_elbow_angle < 40 or right_elbow_angle < 40 and left_elbow_angle>190 or right_elbow_angle > 190:  
    #         cv2.putText(output_image,"Correct your elbows position" , (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            
            
     # Check if the pose is classified successfully
    if label != '':

        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)

    # Write the label on the output image.
    cv2.putText(output_image, label, (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    cv2.putText(output_image, f'Bicep: {bicep_count}', (
                    10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)



    # Check if the resultant image is specified to be displayed.
    if display:

        # Display the resultant image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

    else:

        # Return the output image and the classified label.
        return output_image,label, prev_state, cycle_completed


# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False,
                          min_detection_confidence=0.5, model_complexity=1)

# Initialize the VideoCapture object to read from the webcam.
video = cv2.VideoCapture(0)
# video = cv2.VideoCapture('pushup.mp4')


# Create named window for resizing purposes
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)


# Set video camera size
video.set(3, 1280)
video.set(4, 960)

# Initialize a variable to store the time of the previous frame.
time1 = 0

# Iterate until the video is accessed successfully.
while video.isOpened():
    # Read a frame from the video
    ok, frame = video.read()
    if not ok:
        break

    # Detect pose on the frame
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    frame = cv2.flip(frame, 1)

    # Get the width and height of the frame
    frame_height, frame_width, _ = frame.shape

    # Resize the frame while keeping the aspect ratio.
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

    # Perform Pose landmark detection.
    frame, landmarks = detectPose(frame, pose_video, display=False)

    # Check if the landmarks are detected.
    if landmarks:

        # Perform the Pose Classification.
        frame,label, prev_state, cycle_completed = classifyPose(
            landmarks, frame, prev_state, cycle_completed, display=False)
    
    cv2.imshow('Video', frame)

    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xFF

    # Check if 'ESC' is pressed.
    if (k == 27):

        # Break the loop.
        break

# Release the VideoCapture object and close the windows.
# camera_video.release()
video.release()
cv2.destroyAllWindows()
