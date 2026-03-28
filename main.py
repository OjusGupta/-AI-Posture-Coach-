import cv2
import mediapipe as mp
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
import winsound
import csv
import time

# Global variables
baseline_distance = 0.2  # Default threshold
current_dist = 0
latest_pose_result = None
latest_face_result = None
last_log_time = 0
last_alert_time = 0
log_file = 'posture_log.csv'

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calculate_angle(p1, p2, p3):
    # Angle at p2 between p1-p2-p3
    v1 = (p1.x - p2.x, p1.y - p2.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    cos_angle = dot / (mag1 * mag2)
    angle = math.acos(max(-1, min(1, cos_angle)))
    return math.degrees(angle)

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calculate_angle(p1, p2, p3):
    # Angle at p2 between p1-p2-p3
    v1 = (p1.x - p2.x, p1.y - p2.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    cos_angle = dot / (mag1 * mag2)
    angle = math.acos(max(-1, min(1, cos_angle)))
    return math.degrees(angle)

def pose_callback(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global current_dist, latest_pose_result
    latest_pose_result = result
    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]  # First pose
        # Keypoints: 7 (Left Ear), 11 (Left Shoulder), 12 (Right Shoulder)
        ear = landmarks[7]
        shoulder_l = landmarks[11]
        shoulder_r = landmarks[12]
        current_dist = calculate_distance(ear, shoulder_l)
        # Additional: check shoulder angle or something

def face_callback(result: vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_face_result
    latest_face_result = result

# Create PoseLandmarker
pose_base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    running_mode=vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

# Create FaceLandmarker
face_base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    running_mode=vision.RunningMode.IMAGE,
    min_face_detection_confidence=0.7,
    min_face_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

# Initialize log file
with open(log_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Posture', 'Face', 'Distance', 'Neck_Angle'])

cap = cv2.VideoCapture(0)

print("Press 'c' to calibrate your 'Good Posture' distance.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect pose and face synchronously
    pose_result = pose_landmarker.detect(mp_image)
    face_result = face_landmarker.detect(mp_image)

    # Update globals
    latest_pose_result = pose_result
    latest_face_result = face_result

    # Calculate current_dist and angle
    current_dist = 0
    neck_angle = 180  # Default straight
    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks[0]
        ear = landmarks[7]
        shoulder_l = landmarks[11]
        hip_l = landmarks[23]
        current_dist = calculate_distance(ear, shoulder_l)
        neck_angle = calculate_angle(ear, shoulder_l, hip_l)

    # Posture Logic
    posture_status = "Good Posture"
    posture_color = (0, 255, 0)
    alert_triggered = False

    if current_dist > 0 and (current_dist < (baseline_distance * 0.8) or neck_angle < 160):
        posture_status = "SIT UP STRAIGHT!"
        posture_color = (0, 0, 255)
        current_time = time.time()
        if current_time - last_alert_time > 5:  # Alert every 5 seconds max
            winsound.Beep(1000, 200)  # Audio alert
            last_alert_time = current_time

    # Facial Expression Logic
    face_status = "Neutral"
    face_color = (255, 255, 255)

    if face_result.face_landmarks:
        face_landmarks = face_result.face_landmarks[0]
        # Landmarks
        nose = face_landmarks[1]
        mouth_left = face_landmarks[61]
        mouth_right = face_landmarks[291]
        mouth_top = face_landmarks[13]
        mouth_bottom = face_landmarks[14]
        eye_left_top = face_landmarks[159]
        eye_left_bottom = face_landmarks[145]
        eye_right_top = face_landmarks[386]
        eye_right_bottom = face_landmarks[374]
        brow_left_inner = face_landmarks[70]
        brow_right_inner = face_landmarks[300]

        mouth_open = calculate_distance(mouth_top, mouth_bottom)
        corner_avg_y = (mouth_left.y + mouth_right.y) / 2
        eye_left_open = calculate_distance(eye_left_top, eye_left_bottom)
        eye_right_open = calculate_distance(eye_right_top, eye_right_bottom)
        brow_diff = abs(brow_left_inner.y - brow_right_inner.y)
        eye_distance = calculate_distance(face_landmarks[33], face_landmarks[263])

        if corner_avg_y < nose.y - 0.01 and mouth_open > eye_distance * 0.05:
            face_status = "Smiling"
            face_color = (0, 255, 255)
        elif corner_avg_y > nose.y + 0.01:
            face_status = "Frowning"
            face_color = (255, 0, 255)
        elif mouth_open > eye_distance * 0.1:
            face_status = "Surprised"
            face_color = (255, 255, 0)
        elif brow_diff > 0.05 and (eye_left_open < 0.02 or eye_right_open < 0.02):
            face_status = "Angry"
            face_color = (0, 0, 255)
        elif mouth_open > eye_distance * 0.08 and (eye_left_open < 0.03 or eye_right_open < 0.03):
            face_status = "Tired"
            face_color = (128, 128, 128)

    # Draw UI
    cv2.putText(frame, f"Posture: {posture_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, posture_color, 2)
    cv2.putText(frame, f"Face: {face_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)

    # Draw pose landmarks
    if pose_result.pose_landmarks:
        drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=pose_result.pose_landmarks[0],
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2)
        )

    # Draw face landmarks
    if face_result.face_landmarks:
        drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=face_result.face_landmarks[0],
            connections=None,  # No connections for face
            landmark_drawing_spec=drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
            connection_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=1)
        )

    cv2.imshow('AI Posture Coach with Face Detection', frame)

    # Log data every 10 seconds
    current_time = int(time.time())
    if current_time % 10 == 0 and current_time != last_log_time:
        last_log_time = current_time
        with open(log_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([time.time(), posture_status, face_status, current_dist, neck_angle])

    key = cv2.waitKey(1)
    if key == ord('c'):  # Calibration
        if current_dist > 0:
            baseline_distance = current_dist
            print(f"Calibrated! Baseline: {baseline_distance}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose_landmarker.close()
face_landmarker.close()