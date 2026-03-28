# AI Posture Coach with Facial Expression Detection

An AI-powered tool to monitor posture and facial expressions in real-time using computer vision, designed to help developers prevent slouching and track emotions.

## Features
- **Posture Detection:** Monitors ear-to-shoulder distance and neck angle for slouching alerts with audio feedback.
- **Facial Expressions:** Detects Smiling, Frowning, Surprised, Angry, and Tired using facial landmarks.
- **Real-Time Feedback:** On-screen status, audio beeps for bad posture (every 5 seconds max).
- **Logging:** Saves session data (posture, expressions, metrics) to CSV every 10 seconds for analysis.
- **Calibration:** Press 'c' to set baseline posture.
- **Visualization:** Draws pose (green) and face (red) landmarks on the video feed.

## Installation
1. Ensure Python 3.7+ is installed (download from python.org).
2. Install dependencies:
   ```
   pip install opencv-python mediapipe
   ```
3. Download model files (included in repo or download manually):
   - `pose_landmarker.task`: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
   - `face_landmarker.task`: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

## Usage
1. Run the script:
   ```
   python main.py
   ```
   (Or use full path: `C:/Users/youruser/AppData/Local/Programs/Python/Python311/python.exe main.py`)
2. Allow camera access.
3. Sit straight and press 'c' to calibrate your good posture.
4. View real-time status on screen (posture and face).
5. Press 'q' to quit.
6. Check `posture_log.csv` for logged data (timestamp, status, distance, angle).

## Requirements
- Webcam (built-in or external)
- Python 3.7+
- Libraries: OpenCV, MediaPipe
- OS: Windows (for audio alerts; adapt for others)

## Project Structure
- `main.py`: Main script with detection logic.
- `pose_landmarker.task`: Pose detection model.
- `face_landmarker.task`: Face detection model.
- `posture_log.csv`: Session logs (auto-generated).
- `README.md`: This file.
- `assets/`: Folder for screenshots (add one for submission).

## How It Works
- Uses MediaPipe Pose Landmarker for body keypoints (e.g., ear, shoulder, hip).
- Uses MediaPipe Face Landmarker for facial landmarks (eyes, mouth, brows).
- Calculates distances/angles for posture (e.g., neck angle < 160° triggers alert).
- Analyzes mouth/eye positions for expressions.
- Synchronous detection for real-time performance.

## Example Output
```
Posture: Good Posture
Face: Smiling
```
With pose landmarks (green skeleton) and face landmarks (red dots) overlaid on video.

## Reflection
Initially tried simple Y-coordinate differences for posture, but switched to Euclidean distance and angles for better robustness against movement. Chose Pose Estimation over basic face detection because it's more accurate in varying lighting and poses, handling full-body context.

## Submission Instructions
- **GitHub Repo:** Create a public repo, upload all files, add a screenshot in `assets/`.
- **README:** This file.
- **Report:** PDF with goal, tech used, challenges (API changes), results.
- **Description:** "An AI-powered posture monitoring tool designed for developers to prevent slouching using real-time pose estimation and facial expression analysis."
- **Assets:** Screenshot of the app running.

## License
MIT License.