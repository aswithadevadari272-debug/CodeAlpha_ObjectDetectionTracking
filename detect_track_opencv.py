"""
CodeAlpha Internship - Task 4: Object Detection and Tracking (OpenCV version)
Uses YOLOv4-tiny through OpenCV's DNN module (no PyTorch needed) for detection,
and a simple centroid tracker to give every object an ID across frames.

Usage:
    python detect_track_opencv.py                       # webcam
    python detect_track_opencv.py --source video.mp4    # video file
    python detect_track_opencv.py --source video.mp4 --save
Press 'q' in the video window to quit.
The model files (about 25 MB) download automatically on the first run.
"""

import argparse
import os
import time
import urllib.request
from collections import defaultdict, deque

import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_FILES = {
    "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
    "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
}

PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
]


def download_models():
    """Download the model files if they are not already in the models folder."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, url in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        print(f"Downloading {name} ...")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response, open(path, "wb") as f:
                f.write(response.read())
        except Exception as e:
            if os.path.exists(path):
                os.remove(path)
            raise SystemExit(
                f"Could not download {name}: {e}\n"
                f"Download it manually from:\n{url}\n"
                f"and save it in the folder: {MODEL_DIR}"
            )


class CentroidTracker:
    """Gives each detected object an ID and keeps it while the object moves.
    Detections are matched to existing objects of the same class by the
    smallest distance between box centres."""

    def __init__(self, max_distance=90, max_missing=15):
        self.next_id = 1
        self.points = {}    # id -> (cx, cy)
        self.classes = {}   # id -> class id
        self.missing = {}   # id -> number of frames not seen
        self.max_distance = max_distance
        self.max_missing = max_missing

    def _register(self, point, cls):
        oid = self.next_id
        self.next_id += 1
        self.points[oid] = point
        self.classes[oid] = cls
        self.missing[oid] = 0
        return oid

    def _mark_missing(self, oid):
        self.missing[oid] += 1
        if self.missing[oid] > self.max_missing:
            del self.points[oid], self.classes[oid], self.missing[oid]

    def update(self, centroids, classes):
        """centroids: list of (cx, cy); classes: list of class ids.
        Returns a list of object IDs in the same order as the detections."""
        ids = [None] * len(centroids)
        existing = list(self.points)

        if not centroids:
            for oid in existing:
                self._mark_missing(oid)
            return ids
        if not existing:
            return [self._register(p, c) for p, c in zip(centroids, classes)]

        old_pts = np.array([self.points[o] for o in existing], dtype=float)
        new_pts = np.array(centroids, dtype=float)
        dist = np.linalg.norm(old_pts[:, None, :] - new_pts[None, :, :], axis=2)
        # never match objects of different classes
        for r, oid in enumerate(existing):
            for c, cls in enumerate(classes):
                if self.classes[oid] != cls:
                    dist[r, c] = np.inf

        used_rows, used_cols = set(), set()
        for flat in np.argsort(dist, axis=None):
            r, c = divmod(int(flat), dist.shape[1])
            if dist[r, c] > self.max_distance:
                break  # everything after this is even farther
            if r in used_rows or c in used_cols:
                continue
            oid = existing[r]
            self.points[oid] = centroids[c]
            self.missing[oid] = 0
            ids[c] = oid
            used_rows.add(r)
            used_cols.add(c)

        for r, oid in enumerate(existing):
            if r not in used_rows:
                self._mark_missing(oid)
        for c in range(len(centroids)):
            if c not in used_cols:
                ids[c] = self._register(centroids[c], classes[c])
        return ids


def parse_args():
    parser = argparse.ArgumentParser(description="Object detection and tracking (OpenCV)")
    parser.add_argument("--source", default="0", help="0 for webcam, or a video file path")
    parser.add_argument("--conf", type=float, default=0.4, help="minimum confidence (0 to 1)")
    parser.add_argument("--save", action="store_true", help="save the result as output.mp4")
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    download_models()
    with open(os.path.join(MODEL_DIR, "coco.names"), encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    net = cv2.dnn_DetectionModel(os.path.join(MODEL_DIR, "yolov4-tiny.cfg"),
                                 os.path.join(MODEL_DIR, "yolov4-tiny.weights"))
    net.setInputParams(size=(416, 416), scale=1 / 255.0, swapRB=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open source: {args.source}")
        return

    writer = None
    if args.save:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 20
        writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_in, (width, height))

    tracker = CentroidTracker()
    trails = defaultdict(lambda: deque(maxlen=30))
    seen_ids = set()
    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        class_ids, confs, boxes = net.detect(frame, confThreshold=args.conf,
                                             nmsThreshold=0.4)
        detections = []
        if len(class_ids) > 0:
            for cls, conf, (x, y, w, h) in zip(np.array(class_ids).flatten(),
                                               np.array(confs).flatten(), boxes):
                detections.append((int(cls), float(conf), int(x), int(y), int(w), int(h)))

        centroids = [(x + w // 2, y + h // 2) for _, _, x, y, w, h in detections]
        ids = tracker.update(centroids, [d[0] for d in detections])

        counts = defaultdict(int)
        for (cls, conf, x, y, w, h), track_id, centre in zip(detections, ids, centroids):
            name = names[cls] if cls < len(names) else str(cls)
            counts[name] += 1
            color = PALETTE[track_id % len(PALETTE)]
            seen_ids.add(track_id)
            trails[track_id].append(centre)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"ID {track_id} {name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - th - 8), (x + tw + 4, y), color, -1)
            cv2.putText(frame, label, (x + 2, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            points = trails[track_id]
            for i in range(1, len(points)):  # draw the movement path
                cv2.line(frame, points[i - 1], points[i], color, 2)

        now = time.time()
        fps = 1 / max(now - prev_time, 1e-6)
        prev_time = now
        summary = ", ".join(f"{n}: {c}" for n, c in counts.items()) or "no objects"
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, summary, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Unique objects tracked: {len(seen_ids)}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if writer:
            writer.write(frame)
        cv2.imshow("Object Detection and Tracking - CodeAlpha", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
        print("Saved output.mp4")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
