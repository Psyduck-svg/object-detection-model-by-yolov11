# 🚁 Aerial Guardian – Human Detection & Tracking from Drone Footage

## 📌 Overview

This project implements a real-time **human detection and multi-object tracking pipeline** for aerial (drone) imagery.

The system detects humans in each frame and assigns consistent IDs across frames, enabling tracking and counting in dynamic drone scenarios.

---

## 🎯 Objectives

* Detect humans from aerial imagery
* Track multiple individuals across frames
* Maintain consistent IDs despite motion
* Count number of humans per frame
* Optimize for lightweight and real-time performance

---

## 🧠 Pipeline Architecture

Frame → YOLOv8 Detection → ByteTrack Tracking → Visualization → Output

---

## 🛠️ Technologies Used

* YOLOv8 (Ultralytics) – Object Detection
* ByteTrack – Multi-Object Tracking
* OpenCV – Image Processing
* Python – Implementation

---

## 📂 Dataset

Dataset used:

* VisDrone2019 MOT Validation Set

Structure:

```
VisDrone2019-MOT-val/
├── sequences/
│   ├── uav0000086_00000_v/
│   ├── uav0000339_00001_v/
├── annotations/
```

---

## ⚙️ Installation

Run in Google Colab or local environment:

```bash
pip install ultralytics opencv-python
```

---

## 🚀 How to Run

### Step 1: Load Model

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
```

---

### Step 2: Set Dataset Path

```python
source = "/content/sequences/VisDrone2019-MOT-val/sequences/uav0000086_00000_v"
```

---

### Step 3: Run Tracking

```python
import os
import cv2
from google.colab.patches import cv2_imshow

image_files = sorted(os.listdir(source))

for img_name in image_files:
    img_path = os.path.join(source, img_name)
    frame = cv2.imread(img_path)

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.3,
        tracker="bytetrack.yaml"
    )

    annotated_frame = results[0].plot()
    cv2_imshow(annotated_frame)
```

---

## 👁️ Output

* Bounding boxes around humans
* Unique tracking IDs
* Human count per frame
* Visual output frames

---

## ⚡ Optimization Techniques

### 1. Resolution Scaling

* Reduced input size to 640×480 for faster inference

### 2. Frame Skipping

* Process every 2nd frame to improve speed

### 3. Lightweight Model

* Used YOLOv8n for low computational cost

---

## 🔄 Handling Challenges

### Small Object Detection

* Increased input resolution where needed
* Used YOLOv8 for better feature extraction

### Drone Motion & Noise

* ByteTrack ensures stable tracking under camera movement

### ID Switching

* Persistent tracking (`persist=True`)
* ByteTrack association strategy reduces ID loss

---

## 📊 Performance

* Model: YOLOv8n
* Input Resolution: 640×480
* FPS: ~10–20 FPS (Colab CPU/GPU dependent)

---

## 🚀 Future Improvements

* Add trajectory visualization (motion trails)
* Implement motion compensation (optical flow)
* Fine-tune model on VisDrone dataset
* Deploy on edge devices (NVIDIA Jetson)

---

## 📌 Key Takeaways

* Lightweight models are crucial for drone deployment
* Tracking stability is more important than raw detection accuracy
* Engineering trade-offs (speed vs accuracy) are critical

---
