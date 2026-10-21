"""
AI Posture Coach with Facial Expression Detection
=================================================
A real-time ergonomic posture monitor and facial expression tracker powered by
MediaPipe Pose & Face Landmark detection and OpenCV.

Features:
- Bilateral pose landmark tracking (left & right side fallback)
- Real-time neck angle and ear-to-shoulder euclidean distance computation
- Facial expression emotion detection (Smiling, Frowning, Surprised, Angry, Tired, Neutral)
- Glassmorphic translucent HUD overlay with live ergonomics metrics & score
- Audio slouch alerts with mute toggle
- Dynamic baseline calibration ('C') and wireframe visibility toggle ('L')
- Session logging with ISO timestamps to CSV
"""

import os
import sys
import time
import math
import csv
from datetime import datetime
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils

# Platform-safe audio alert
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


# =====================================================================
# Configuration & Constants
# =====================================================================
DEFAULT_BASELINE_DISTANCE = 0.20  # Default normalized ear-shoulder distance
SLOUCH_DISTANCE_RATIO = 0.80      # Trigger alert if distance < 80% of baseline
MIN_NECK_ANGLE = 160.0            # Trigger alert if neck angle < 160 deg
ALERT_COOLDOWN_SEC = 5.0          # Minimum seconds between audio beeps
LOG_INTERVAL_SEC = 10.0           # Seconds between CSV log entries
CSV_LOG_PATH = "posture_log.csv"
POSE_MODEL_PATH = "pose_landmarker.task"
FACE_MODEL_PATH = "face_landmarker.task"


# =====================================================================
# Mathematical & Geometric Helpers
# =====================================================================
def calculate_distance(p1, p2):
    """Calculate 2D Euclidean distance between two normalized landmarks."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def calculate_angle(p1, p2, p3):
    """
    Calculate the angle in degrees at vertex p2 formed by lines (p1-p2) and (p3-p2).
    """
    v1 = (p1.x - p2.x, p1.y - p2.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])

    if mag1 * mag2 == 0:
        return 180.0

    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def play_alert_sound():
    """Emit an audio alert safely across platforms."""
    if HAS_WINSOUND:
        try:
            winsound.Beep(1000, 250)
        except Exception:
            pass
    else:
        sys.stdout.write("\a")
        sys.stdout.flush()


# =====================================================================
# UI Drawing Utilities (Glassmorphic HUD)
# =====================================================================
def draw_rounded_rect(img, pt1, pt2, color, thickness=1, radius=10):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)

    # Corners
    cv2.circle(img, (x1 + r, y1 + r), r, color, thickness)
    cv2.circle(img, (x2 - r, y1 + r), r, color, thickness)
    cv2.circle(img, (x1 + r, y2 - r), r, color, thickness)
    cv2.circle(img, (x2 - r, y2 - r), r, color, thickness)

    # Straight edges
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)


def draw_hud_panel(frame, posture_status, face_status, current_dist, baseline_dist,
                   neck_angle, fps, posture_score, slouch_count, is_muted,
                   is_calibrated, session_time_str):
    """
    Renders a futuristic translucent glassmorphic HUD overlay with real-time metrics.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # Panel dimensions
    px, py = 16, 16
    pw, ph = 340, 390

    # Draw semi-transparent dark HUD card
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Panel border accent
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (60, 75, 95), 1)

    # 1. Header: Title & Live FPS / Timer
    cv2.putText(frame, "AI POSTURE COACH", (px + 14, py + 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 230, 255), 1, cv2.LINE_AA)
    meta_text = f"FPS: {fps:.0f} | {session_time_str}"
    cv2.putText(frame, meta_text, (px + pw - 135, py + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 185, 200), 1, cv2.LINE_AA)
    cv2.line(frame, (px + 14, py + 36), (px + pw - 14, py + 36), (50, 65, 85), 1)

    # 2. Primary Posture Status Badge
    is_good = (posture_status == "Good Posture")
    badge_bg = (40, 160, 80) if is_good else (40, 45, 215)
    badge_label = "GOOD POSTURE" if is_good else "SLOUCH DETECTED!"
    if posture_status == "Calibrating...":
        badge_bg = (30, 140, 210)
        badge_label = "CALIBRATING..."
    elif posture_status == "No Person":
        badge_bg = (70, 80, 95)
        badge_label = "NO PERSON DETECTED"

    # Pill badge
    bx1, by1 = px + 14, py + 48
    bx2, by2 = px + pw - 14, py + 84
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), badge_bg, -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
    cv2.putText(frame, badge_label, (bx1 + 18, by1 + 24),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    # 3. Facial Expression Badge
    ey = py + 104
    cv2.putText(frame, "Facial Expression:", (px + 14, ey + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 195, 210), 1, cv2.LINE_AA)
    face_color = (0, 240, 255) if face_status == "Smiling" else (
        (255, 120, 220) if face_status == "Frowning" else (
            (255, 220, 80) if face_status == "Surprised" else (
                (80, 80, 255) if face_status == "Angry" else (
                    (180, 180, 180) if face_status == "Tired" else (220, 230, 240)
                )
            )
        )
    )
    cv2.putText(frame, f"[{face_status.upper()}]", (px + 160, ey + 14),
                cv2.FONT_HERSHEY_DUPLEX, 0.50, face_color, 1, cv2.LINE_AA)

    cv2.line(frame, (px + 14, ey + 24), (px + pw - 14, ey + 24), (45, 58, 75), 1)

    # 4. Ergonomics Telemetry Section
    ty = ey + 42
    # Neck Angle
    angle_color = (100, 240, 140) if neck_angle >= MIN_NECK_ANGLE else (80, 100, 255)
    cv2.putText(frame, f"Neck Angle: {neck_angle:.1f} deg", (px + 14, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 230, 240), 1, cv2.LINE_AA)
    cv2.putText(frame, f"(Min: {MIN_NECK_ANGLE:.0f} deg)", (px + pw - 105, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, angle_color, 1, cv2.LINE_AA)

    # Neck progress bar
    bar_y = ty + 8
    cv2.rectangle(frame, (px + 14, bar_y), (px + pw - 14, bar_y + 8), (40, 50, 65), -1)
    norm_angle = max(0.0, min(1.0, (neck_angle - 120.0) / 60.0))
    bar_w = int((pw - 28) * norm_angle)
    cv2.rectangle(frame, (px + 14, bar_y), (px + 14 + bar_w, bar_y + 8), angle_color, -1)

    # Ear-to-Shoulder Distance
    dy = bar_y + 28
    dist_ratio = (current_dist / baseline_dist * 100.0) if baseline_dist > 0 else 100.0
    dist_color = (100, 240, 140) if dist_ratio >= (SLOUCH_DISTANCE_RATIO * 100) else (80, 100, 255)
    cv2.putText(frame, f"Ear-Shoulder Dist: {current_dist:.3f}", (px + 14, dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 230, 240), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{dist_ratio:.0f}% Base", (px + pw - 90, dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, dist_color, 1, cv2.LINE_AA)

    # Distance bar
    dbar_y = dy + 8
    cv2.rectangle(frame, (px + 14, dbar_y), (px + pw - 14, dbar_y + 8), (40, 50, 65), -1)
    norm_dist = max(0.0, min(1.0, dist_ratio / 130.0))
    dbar_w = int((pw - 28) * norm_dist)
    cv2.rectangle(frame, (px + 14, dbar_y), (px + 14 + dbar_w, dbar_y + 8), dist_color, -1)

    cv2.line(frame, (px + 14, dbar_y + 18), (px + pw - 14, dbar_y + 18), (45, 58, 75), 1)

    # 5. Session Ergonomics Score & Alerts
    sy = dbar_y + 36
    score_color = (80, 230, 130) if posture_score >= 80 else (
        (240, 190, 60) if posture_score >= 60 else (80, 90, 255)
    )
    cv2.putText(frame, f"Posture Score: {posture_score:.1f}%", (px + 14, sy),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, score_color, 1, cv2.LINE_AA)

    cv2.putText(frame, f"Slouch Alerts: {slouch_count}", (px + 14, sy + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 215, 230), 1, cv2.LINE_AA)

    calib_str = f"Base: {baseline_dist:.3f}" if is_calibrated else "Default Base"
    audio_str = "Audio: MUTED" if is_muted else "Audio: ON"
    cv2.putText(frame, f"{calib_str} | {audio_str}", (px + 14, sy + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 170, 190), 1, cv2.LINE_AA)

    # 6. Controls Footer Bar
    cv2.line(frame, (px + 14, sy + 54), (px + pw - 14, sy + 54), (50, 65, 85), 1)
    cv2.putText(frame, "[C] Calibrate  [L] Mesh  [M] Mute  [Q] Quit", (px + 14, sy + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (130, 210, 240), 1, cv2.LINE_AA)


# =====================================================================
# Main Application Pipeline
# =====================================================================
def main():
    print("==================================================")
    print("  AI Posture Coach with Facial Expression Tracking")
    print("==================================================")
    print("Initializing MediaPipe models...")

    # Verify model files
    if not os.path.exists(POSE_MODEL_PATH) or not os.path.exists(FACE_MODEL_PATH):
        print(f"Error: Model files '{POSE_MODEL_PATH}' or '{FACE_MODEL_PATH}' not found!")
        print("Please verify the model files exist in the project directory.")
        return

    # Initialize Pose Landmarker
    pose_base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    # Initialize Face Landmarker
    face_base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

    # Initialize CSV Log file (append mode, writes header if empty)
    file_exists = os.path.exists(CSV_LOG_PATH) and os.path.getsize(CSV_LOG_PATH) > 0
    with open(CSV_LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'Timestamp', 'Session_Elapsed_Sec', 'Posture_Status',
                'Face_Status', 'Current_Dist', 'Baseline_Dist',
                'Neck_Angle', 'Posture_Score_Pct', 'Slouch_Alert_Count'
            ])

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera (device index 0).")
        return

    # Application state variables
    baseline_distance = DEFAULT_BASELINE_DISTANCE
    is_calibrated = False
    is_muted = False
    show_mesh = True
    last_alert_time = 0.0
    last_log_time = 0.0
    session_start_time = time.time()
    slouch_count = 0
    total_frames = 0
    good_frames = 0

    fps_tracker_time = time.time()
    frame_count = 0
    fps = 30.0

    print("Camera feed active.")
    print("Press 'c' to Calibrate baseline good posture.")
    print("Press 'l' to Toggle skeleton & facial landmark mesh.")
    print("Press 'm' to Toggle audio alert sound.")
    print("Press 'r' to Reset session metrics.")
    print("Press 'q' to Quit application.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Warning: Failed to capture video frame. Exiting...")
            break

        total_frames += 1
        frame_count += 1

        # Calculate FPS
        current_time = time.time()
        if current_time - fps_tracker_time >= 1.0:
            fps = frame_count / (current_time - fps_tracker_time)
            frame_count = 0
            fps_tracker_time = current_time

        # Format session elapsed time
        elapsed_sec = int(current_time - session_start_time)
        session_time_str = f"{elapsed_sec // 60:02d}:{elapsed_sec % 60:02d}"

        # Mirror horizontally for natural interactive feel
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect pose and face synchronously
        pose_result = pose_landmarker.detect(mp_image)
        face_result = face_landmarker.detect(mp_image)

        # =============================================================
        # 1. Posture Estimation Logic (Bilateral Keypoint Support)
        # =============================================================
        current_dist = 0.0
        neck_angle = 180.0
        posture_status = "No Person"

        if pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks[0]

            # Left side keypoints: 7 (ear), 11 (shoulder), 23 (hip)
            # Right side keypoints: 8 (ear), 12 (shoulder), 24 (hip)
            left_ear, left_shoulder = landmarks[7], landmarks[11]
            right_ear, right_shoulder = landmarks[8], landmarks[12]

            left_hip = landmarks[23] if len(landmarks) > 23 else None
            right_hip = landmarks[24] if len(landmarks) > 24 else None

            # Choose side with higher landmark presence/visibility
            left_score = getattr(left_shoulder, 'presence', 0.5) + getattr(left_ear, 'presence', 0.5)
            right_score = getattr(right_shoulder, 'presence', 0.5) + getattr(right_ear, 'presence', 0.5)

            if left_score >= right_score:
                ear, shoulder, hip = left_ear, left_shoulder, left_hip
            else:
                ear, shoulder, hip = right_ear, right_shoulder, right_hip

            # Euclidean distance between ear and shoulder
            current_dist = calculate_distance(ear, shoulder)

            # Neck angle (Ear - Shoulder - Hip)
            if hip is not None and getattr(hip, 'visibility', 0.5) > 0.2:
                neck_angle = calculate_angle(ear, shoulder, hip)
            else:
                # Fallback: estimate inclination against vertical trunk axis
                virtual_hip_x = shoulder.x
                virtual_hip_y = shoulder.y + 0.35
                v_hip = type('Point', (), {'x': virtual_hip_x, 'y': virtual_hip_y})()
                neck_angle = calculate_angle(ear, shoulder, v_hip)

            # Determine posture status
            is_slouching = (
                (current_dist > 0 and current_dist < (baseline_distance * SLOUCH_DISTANCE_RATIO)) or
                (neck_angle < MIN_NECK_ANGLE)
            )

            if is_slouching:
                posture_status = "SIT UP STRAIGHT!"
                if current_time - last_alert_time >= ALERT_COOLDOWN_SEC:
                    slouch_count += 1
                    if not is_muted:
                        play_alert_sound()
                    last_alert_time = current_time
            else:
                posture_status = "Good Posture"
                good_frames += 1

        # Posture score calculation (% of frames with good posture)
        posture_score = (good_frames / total_frames * 100.0) if total_frames > 0 else 100.0

        # =============================================================
        # 2. Facial Expression Logic
        # =============================================================
        face_status = "Neutral"

        if face_result.face_landmarks:
            fl = face_result.face_landmarks[0]

            nose = fl[1]
            mouth_left = fl[61]
            mouth_right = fl[291]
            mouth_top = fl[13]
            mouth_bottom = fl[14]
            eye_left_top = fl[159]
            eye_left_bottom = fl[145]
            eye_right_top = fl[386]
            eye_right_bottom = fl[374]
            brow_left_inner = fl[70]
            brow_right_inner = fl[300]

            mouth_open = calculate_distance(mouth_top, mouth_bottom)
            corner_avg_y = (mouth_left.y + mouth_right.y) / 2.0
            eye_left_open = calculate_distance(eye_left_top, eye_left_bottom)
            eye_right_open = calculate_distance(eye_right_top, eye_right_bottom)
            brow_diff = abs(brow_left_inner.y - brow_right_inner.y)
            eye_distance = calculate_distance(fl[33], fl[263])

            if corner_avg_y < nose.y - 0.01 and mouth_open > (eye_distance * 0.05):
                face_status = "Smiling"
            elif corner_avg_y > nose.y + 0.01:
                face_status = "Frowning"
            elif mouth_open > (eye_distance * 0.10):
                face_status = "Surprised"
            elif brow_diff > 0.05 and (eye_left_open < 0.02 or eye_right_open < 0.02):
                face_status = "Angry"
            elif mouth_open > (eye_distance * 0.08) and (eye_left_open < 0.03 or eye_right_open < 0.03):
                face_status = "Tired"

        # =============================================================
        # 3. Landmark Visualizations (Toggleable)
        # =============================================================
        if show_mesh:
            # Pose Skeleton
            if pose_result.pose_landmarks:
                drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=pose_result.pose_landmarks[0],
                    connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                    landmark_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=2),
                    connection_drawing_spec=drawing_utils.DrawingSpec(color=(0, 230, 255), thickness=2)
                )

            # Facial Mesh Dots
            if face_result.face_landmarks:
                drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=face_result.face_landmarks[0],
                    connections=None,
                    landmark_drawing_spec=drawing_utils.DrawingSpec(color=(255, 70, 70), thickness=1, circle_radius=1),
                    connection_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=1)
                )

        # =============================================================
        # 4. Render Glassmorphic Telemetry HUD Overlay
        # =============================================================
        draw_hud_panel(
            frame=frame,
            posture_status=posture_status,
            face_status=face_status,
            current_dist=current_dist,
            baseline_dist=baseline_distance,
            neck_angle=neck_angle,
            fps=fps,
            posture_score=posture_score,
            slouch_count=slouch_count,
            is_muted=is_muted,
            is_calibrated=is_calibrated,
            session_time_str=session_time_str
        )

        cv2.imshow("AI Posture Coach with Facial Expression Tracking", frame)

        # =============================================================
        # 5. Periodic CSV Logging (every 10 seconds)
        # =============================================================
        if current_time - last_log_time >= LOG_INTERVAL_SEC:
            last_log_time = current_time
            now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                with open(CSV_LOG_PATH, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        now_iso, elapsed_sec, posture_status,
                        face_status, f"{current_dist:.4f}", f"{baseline_distance:.4f}",
                        f"{neck_angle:.2f}", f"{posture_score:.1f}", slouch_count
                    ])
            except Exception as e:
                print(f"Warning: Failed to write to CSV log: {e}")

        # =============================================================
        # 6. Interactive Keyboard Event Handling
        # =============================================================
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('c'), ord('C')):  # Calibrate Baseline
            if current_dist > 0:
                baseline_distance = current_dist
                is_calibrated = True
                print(f"[Calibration] New baseline distance set: {baseline_distance:.4f}")

        elif key in (ord('l'), ord('L')):  # Toggle Skeleton / Mesh
            show_mesh = not show_mesh
            print(f"[Mesh Toggle] Landmark mesh visibility: {'ON' if show_mesh else 'OFF'}")

        elif key in (ord('m'), ord('M')):  # Mute / Unmute Audio
            is_muted = not is_muted
            print(f"[Audio Toggle] Audio alert: {'MUTED' if is_muted else 'ACTIVE'}")

        elif key in (ord('r'), ord('R')):  # Reset Session
            total_frames = 0
            good_frames = 0
            slouch_count = 0
            session_start_time = time.time()
            print("[Session] Ergonomics statistics reset.")

        elif key in (ord('q'), ord('Q')):  # Quit
            print("[Quit] Exiting AI Posture Coach. Good job maintaining posture!")
            break

    # Resource Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose_landmarker.close()
    face_landmarker.close()


if __name__ == "__main__":
    main()