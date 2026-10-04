# Object Detection and Tracking

**CodeAlpha AI Internship – Task 4**

Real-time object detection and tracking from a webcam or a video file. It uses a pretrained YOLOv8 model to detect objects, ByteTrack to give every object a persistent ID, and OpenCV to draw the results.

## Features
- Detects 80 everyday object types (person, car, bottle, phone, etc.)
- Tracks each object with a unique ID across frames
- Draws a movement trail behind every tracked object
- Shows live FPS, objects per class, and the number of unique objects tracked
- Works with a webcam or any video file, and can save the result as `output.mp4`

## Tech Stack
- Python 3.9 – 3.12 recommended
- [Ultralytics YOLOv8](https://docs.ultralytics.com/) (detection + ByteTrack tracking)
- OpenCV (video input and drawing)

## How to Run
```bash
git clone https://github.com/<your-username>/CodeAlpha_ObjectDetectionTracking.git
cd CodeAlpha_ObjectDetectionTracking
pip install -r requirements.txt

python detect_track.py                            # webcam
python detect_track.py --source video.mp4         # video file
python detect_track.py --source video.mp4 --save  # also save output.mp4
```
Press **q** in the video window to quit. The YOLO model (`yolov8n.pt`) downloads automatically on the first run, so an internet connection is needed once.

## Options
| Option | Meaning | Default |
|---|---|---|
| `--source` | `0` for webcam, or a video file path | `0` |
| `--model` | YOLO model file | `yolov8n.pt` |
| `--conf` | minimum confidence (0–1) | `0.4` |
| `--save` | save the output video | off |

## How It Works
1. Each frame is read with OpenCV.
2. YOLOv8 detects objects and returns boxes, classes and confidence scores.
3. ByteTrack matches detections between frames and assigns each object an ID.
4. The script draws boxes, ID labels and movement trails, then shows the frame.

## Author
Aswitha Devadari

## Lightweight version (no PyTorch)
If PyTorch/ultralytics can't be installed (for example on a very new Python version), use `detect_track_opencv.py`. It runs YOLOv4-tiny through OpenCV's DNN module and tracks objects with a simple centroid tracker (nearest box centre, same class).

```bash
pip install opencv-python numpy
python detect_track_opencv.py                       # webcam
python detect_track_opencv.py --source video.mp4 --save
```
The model files (about 25 MB) download automatically into a `models` folder on the first run.
