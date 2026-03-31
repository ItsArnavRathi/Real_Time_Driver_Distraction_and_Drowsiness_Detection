# Real-Time Driver Distraction and Drowsiness Detection

A computer vision based driver monitoring system that detects drowsiness and fatigue in real time using facial landmark analysis. Built with MediaPipe and OpenCV, the system continuously analyzes the driver's face through a webcam and triggers an audio alert when unsafe conditions are detected.

---

## Problem Statement

Driver fatigue and distraction are among the leading causes of road accidents. Most vehicles do not have any system that monitors the driver's alertness during a journey. This project addresses that gap by providing a real-time, vision-based solution that works on standard hardware without any specialized equipment.

---

## Features

- Eye Aspect Ratio (EAR) based drowsiness detection
- Mouth Aspect Ratio (MAR) based yawn detection
- Frame-counter logic to avoid false positives from momentary blinks
- Continuous audio alarm via `pygame.mixer` that starts and stops in sync with the alert condition
- Live HUD showing EAR and MAR values on the video feed
- MediaPipe face landmarker model auto-downloads on first run

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Language | Python 3.9+ |
| Video Capture | OpenCV (`cv2`) |
| Face Landmarks | MediaPipe FaceLandmarker |
| Feature Extraction | NumPy |
| Audio Alert | pygame |

---

## Project Structure

```
Real_Time_Driver_Distraction_and_Drowsiness_Detection/
    main.py                   # Entry point - run this to start the system
    alarm.mp3                 # Alert sound file (add your own)
    face_landmarker.task      # MediaPipe model (auto-downloaded on first run)
    requirements.txt          # Python dependencies
    README.md
```

---


## How It Works

**Eye Aspect Ratio (EAR)**

Six landmark points around each eye are used to compute a ratio that reflects how open the eye is. When the driver's eye closes, the EAR drops below the threshold. If it stays below that threshold for more than `FRAME_THRESHOLD` consecutive frames, a drowsiness alert is triggered.

```
EAR = (vertical_dist_1 + vertical_dist_2) / (2 * horizontal_dist)
```

**Mouth Aspect Ratio (MAR)**

A similar ratio is computed for the mouth. A high MAR value sustained over multiple frames indicates yawning, which is treated as a fatigue signal.

**Alert Logic**

When either condition is active, `pygame.mixer` starts looping the alarm. The moment the condition clears, the alarm stops immediately. This ensures there is no overlap or delayed stopping of the sound.

---

## Configurable Parameters

These values are defined at the top of `main.py` and can be tuned based on your camera distance and lighting:

| Parameter | Default | Description |
|---|---|---|
| `EAR_THRESHOLD` | `0.20` | EAR below this = eye considered closed |
| `FRAME_THRESHOLD` | `15` | Consecutive frames before drowsy alert fires |
| `MAR_THRESHOLD` | `0.70` | MAR above this = mouth considered open (yawn) |
| `YAWN_FRAMES` | `15` | Consecutive frames before yawn alert fires |

---

## Requirements

```
opencv-python
mediapipe
numpy
pygame
```

Install all at once:

```bash
pip install opencv-python mediapipe numpy pygame
```

---

## Landmark Indices Used

| Region | MediaPipe Indices |
|---|---|
| Left Eye | 33, 160, 158, 133, 153, 144 |
| Right Eye | 362, 385, 387, 263, 373, 380 |
| Mouth | 13, 14, 78, 308, 82, 312 |

---

## Limitations

- Works best in well-lit conditions; low light affects landmark accuracy
- Single face detection only (designed for a single driver)
- Does not currently detect phone usage (planned as next module using CNN)
- Head pose is not factored in; looking sideways may give inaccurate EAR readings

---

## Planned Additions

- Mobile phone detection using MobileNetV2 transfer learning
- Head pose estimation to detect distracted gaze direction
- Time-based distraction logging with session history
- Simple GUI dashboard for monitoring status

---

## References

- Soukupova, T. and Cech, J. — *Real-Time Eye Blink Detection using Facial Landmarks*, CVWW 2016
- [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- [OpenCV Documentation](https://docs.opencv.org)
- [pygame Documentation](https://www.pygame.org/docs)
