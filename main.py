import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import numpy as np
import urllib.request
import os
import pygame

# -------------------------------
# Initialize Sound
# -------------------------------
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.mp3")

# -------------------------------
# Load YOLO Model (Phone Detection)
# -------------------------------
yolo_model = YOLO("yolov8n.pt")

# -------------------------------
# Download Face Model if needed
# -------------------------------
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        model_path
    )
    print("Download complete.")

# -------------------------------
# Initialize Face Landmarker
# -------------------------------
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# -------------------------------
# Landmark Indices
# -------------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308, 82, 312]

# -------------------------------
# EAR Function
# -------------------------------
def calculate_ear(landmarks, eye_indices, w, h):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]

    v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    h_dist = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    return (v1 + v2) / (2.0 * h_dist)

# -------------------------------
# MAR Function
# -------------------------------
def calculate_mar(landmarks, mouth_indices, w, h):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in mouth_indices]

    v_dist = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
    h_dist = np.linalg.norm(np.array(points[2]) - np.array(points[3]))

    return v_dist / h_dist

# -------------------------------
# Thresholds
# -------------------------------
EAR_THRESHOLD = 0.20
FRAME_THRESHOLD = 15

MAR_THRESHOLD = 0.7
YAWN_FRAMES = 15

PHONE_FRAMES = 10

# -------------------------------
# Counters & Flags
# -------------------------------
eye_counter = 0
yawn_counter = 0
phone_counter = 0
alarm_on = False

# -------------------------------
# Start Webcam
# -------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------------
    # YOLO Phone Detection (resize for speed)
    # -------------------------------
    small_frame = cv2.resize(frame, (640, 480))
    results = yolo_model(small_frame, verbose=False)

    phone_detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = yolo_model.names[cls]

            if label == "cell phone":
                phone_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, "PHONE", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    if phone_detected:
        phone_counter += 1
    else:
        phone_counter = 0

    # -------------------------------
    # Face Detection
    # -------------------------------
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    alert_triggered = False

    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:
            h, w, _ = frame.shape

            # EAR
            left_ear = calculate_ear(face_landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(face_landmarks, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if ear < EAR_THRESHOLD:
                eye_counter += 1
            else:
                eye_counter = 0

            # MAR
            mar = calculate_mar(face_landmarks, MOUTH, w, h)

            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if mar > MAR_THRESHOLD:
                yawn_counter += 1
            else:
                yawn_counter = 0

            # Alerts
            if eye_counter > FRAME_THRESHOLD:
                alert_triggered = True
                cv2.putText(frame, "DROWSY!", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            if yawn_counter > YAWN_FRAMES:
                alert_triggered = True
                cv2.putText(frame, "YAWNING!", (30, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            # Draw landmarks
            for landmark in face_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Phone Alert
    if phone_counter > PHONE_FRAMES:
        alert_triggered = True
        cv2.putText(frame, "PHONE DETECTED!", (30, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

    # Final Alert + Sound
    if alert_triggered:
        cv2.putText(frame, "ALERT!", (30, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

        if not alarm_on:
            alarm_on = True
            alarm_sound.play(-1)
    else:
        if alarm_on:
            alarm_on = False
            alarm_sound.stop()

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()