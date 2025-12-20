# Vision Setup Guide

Configure camera and brick detection for LEGO MCP Studio.

---

## Overview

The vision system enables automatic brick detection using:
- **YOLO11**: Fast local detection (recommended)
- **Roboflow**: Cloud-based detection (no GPU needed)
- **Mock**: Testing without camera

---

## Quick Setup

### Option 1: YOLO11 (Recommended)

Best for: Fast detection, offline use, privacy

```bash
pip install ultralytics opencv-python numpy pillow
```

That's it! YOLO11 will auto-download models on first use.

### Option 2: Roboflow

Best for: No GPU, cloud processing, pre-trained models

```bash
pip install roboflow inference-sdk
```

Set your API key:
```bash
export ROBOFLOW_API_KEY=your_key_here
```

Get a key at [roboflow.com](https://roboflow.com)

### Option 3: Mock (Testing)

Best for: Development, no camera available

```bash
export DETECTION_BACKEND=mock
```

No additional installation needed.

---

## Camera Setup

### List Available Cameras

```python
import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i}: Available")
        cap.release()
```

### Configure Camera

In your `.env` file:

```bash
CAMERA_INDEX=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=30
```

### Webcam Tips

- Position camera directly above baseplate
- Ensure even lighting (no shadows)
- Use a plain background
- Keep camera steady (tripod recommended)

---

## Detection Backends

### YOLO11 (Local)

**Installation:**
```bash
pip install ultralytics
```

**Configuration:**
```bash
DETECTION_BACKEND=yolo
YOLO_MODEL=yolov8n  # Options: yolov8n, yolov8s, yolov8m, yolov8l
DETECTION_THRESHOLD=0.5
```

**Models:**
| Model | Speed | Accuracy | Size |
|-------|-------|----------|------|
| yolov8n | Fastest | Good | 6 MB |
| yolov8s | Fast | Better | 22 MB |
| yolov8m | Medium | Great | 52 MB |
| yolov8l | Slow | Best | 87 MB |

### Roboflow (Cloud)

**Installation:**
```bash
pip install roboflow inference-sdk
```

**Configuration:**
```bash
DETECTION_BACKEND=roboflow
ROBOFLOW_API_KEY=your_key
ROBOFLOW_MODEL=lego-bricks/1
DETECTION_THRESHOLD=0.5
```

**Getting API Key:**
1. Create account at [roboflow.com](https://roboflow.com)
2. Go to Settings → API Keys
3. Copy your API key

### Mock (Testing)

**Configuration:**
```bash
DETECTION_BACKEND=mock
```

Mock detector generates random bricks for testing the UI.

---

## Color Classification

The system identifies 43 official LEGO colors:

### Primary Colors
| Color | RGB | Hex |
|-------|-----|-----|
| Red | 201, 26, 9 | #C91A09 |
| Blue | 0, 87, 168 | #0057A8 |
| Yellow | 247, 209, 23 | #F7D117 |
| Green | 0, 133, 43 | #00852B |
| Black | 27, 42, 52 | #1B2A34 |
| White | 255, 255, 255 | #FFFFFF |

### Extended Colors
| Color | RGB |
|-------|-----|
| Orange | 254, 138, 24 |
| Light Gray | 160, 165, 169 |
| Dark Gray | 91, 103, 112 |
| Brown | 88, 57, 39 |
| Tan | 228, 205, 158 |
| Dark Blue | 0, 29, 104 |
| Dark Green | 0, 100, 46 |
| Dark Red | 114, 0, 18 |
| Pink | 252, 151, 172 |
| Purple | 104, 37, 138 |
| Lime | 166, 202, 85 |
| Medium Azure | 66, 192, 251 |
| Dark Turquoise | 0, 138, 128 |
| Bright Light Orange | 248, 187, 61 |
| Trans Clear | 252, 252, 252 |
| Trans Red | 201, 26, 9 |
| Trans Blue | 0, 87, 168 |

---

## Calibration

### Baseplate Calibration

For accurate grid mapping:

1. Go to Workspace page
2. Click **Calibrate**
3. Click the 4 corners of your baseplate (in order):
   - Top-left
   - Top-right
   - Bottom-right
   - Bottom-left
4. Click **Save Calibration**

### Region of Interest (ROI)

Limit detection to a specific area:

```json
{
  "roi_x1": 100,
  "roi_y1": 50,
  "roi_x2": 1180,
  "roi_y2": 670
}
```

Set via `/workspace/config` API or Settings page.

---

## Workspace Configuration

### Full Configuration

```json
{
  "grid_size": 8,
  "roi_x1": 0,
  "roi_y1": 0,
  "roi_x2": 1280,
  "roi_y2": 720,
  "detection_threshold": 0.5,
  "stability_threshold": 3,
  "stability_time_seconds": 1.0,
  "corners": [
    [100, 100],
    [1100, 100],
    [1100, 600],
    [100, 600]
  ]
}
```

| Setting | Description | Default |
|---------|-------------|---------|
| grid_size | Grid dimensions (NxN) | 8 |
| roi_* | Region of interest | Full frame |
| detection_threshold | Min confidence | 0.5 |
| stability_threshold | Frames before stable | 3 |
| stability_time_seconds | Time before stable | 1.0 |
| corners | Baseplate calibration | None |

---

## Troubleshooting

### No Cameras Detected

**Check permissions (Linux):**
```bash
sudo usermod -aG video $USER
# Then log out and back in
```

**Check device exists:**
```bash
ls /dev/video*
```

**Use mock camera:**
```bash
export DETECTION_BACKEND=mock
```

### Poor Detection Accuracy

**Improve lighting:**
- Use diffuse, even lighting
- Avoid direct sunlight
- No shadows on bricks

**Increase confidence threshold:**
```bash
DETECTION_THRESHOLD=0.7
```

**Use better model:**
```bash
YOLO_MODEL=yolov8m  # Medium model
```

### Slow Detection

**Use lighter model:**
```bash
YOLO_MODEL=yolov8n  # Nano model
```

**Reduce resolution:**
```bash
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
```

**Use GPU (if available):**
```bash
pip install torch torchvision  # CUDA version
```

### Color Misidentification

Colors are classified by closest RGB match. Tips:
- Use consistent white-balanced lighting
- Avoid colored light sources
- Clean bricks (dust affects color)

### Camera Feed Freezes

**Lower FPS:**
```bash
CAMERA_FPS=15
```

**Check USB bandwidth:**
- Use USB 3.0 port
- Disconnect other USB devices

---

## API Reference

### Detector Info
```
GET /workspace/detector/info
```

**Response:**
```json
{
  "backend": "yolo",
  "model": "yolov8n",
  "confidence_threshold": 0.5,
  "yolo_available": true,
  "cv2_available": true,
  "roboflow_available": false
}
```

### Run Detection
```
POST /workspace/detect
```

### Camera Info
```
GET /workspace/cameras
```

---

## Hardware Recommendations

### Cameras

| Camera | Price | Quality | Notes |
|--------|-------|---------|-------|
| Logitech C920 | $70 | Good | Great value |
| Logitech Brio | $130 | Excellent | 4K |
| Razer Kiyo | $80 | Good | Built-in light |
| Any 1080p webcam | $30+ | Adequate | Position matters more |

### Lighting

| Solution | Price | Notes |
|----------|-------|-------|
| Ring light | $20-50 | Easy even lighting |
| LED panel | $30-100 | Professional results |
| Desk lamp | $15 | Works in a pinch |

### Mounting

| Solution | Notes |
|----------|-------|
| Tripod | Most flexible |
| Monitor mount arm | Stable overhead |
| DIY bracket | Custom positioning |

---

## Advanced: Custom Models

### Training Custom YOLO Model

For improved accuracy, train on your own bricks:

1. Collect images of your bricks
2. Label with [Roboflow](https://roboflow.com)
3. Export in YOLO format
4. Train:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='your_dataset.yaml', epochs=100)
```

5. Use custom model:

```bash
YOLO_MODEL=/path/to/best.pt
```

### Using Roboflow Custom Model

1. Train model at Roboflow
2. Deploy to API
3. Configure:

```bash
ROBOFLOW_MODEL=your-project/version
```

---

## Performance Benchmarks

Tested on various hardware:

| Hardware | Model | FPS |
|----------|-------|-----|
| MacBook Pro M1 | yolov8n | 30+ |
| MacBook Pro M1 | yolov8m | 15 |
| RTX 3080 | yolov8n | 60+ |
| RTX 3080 | yolov8l | 30+ |
| Intel i7 (no GPU) | yolov8n | 8-10 |
| Raspberry Pi 4 | yolov8n | 2-3 |
| Roboflow Cloud | - | 5-10 |

---

## Next Steps

- [Workspace Guide](USER_GUIDE.md#workspace-digital-twin)
- [Scanning Tutorial](USER_GUIDE.md#scanning-bricks)
- [API Reference](API.md#workspace-api)
