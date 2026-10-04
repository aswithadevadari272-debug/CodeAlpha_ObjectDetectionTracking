"""
CodeAlpha Internship - Task 4: Object Detection and Tracking
Real-time object detection with YOLOv8 and tracking with ByteTrack,
displayed with OpenCV. Works with a webcam or a video file.

Usage:
    python detect_track.py                          # webcam
    python detect_track.py --source video.mp4       # video file
    python detect_track.py --source video.mp4 --save   # also save output.mp4
Press 'q' in the video window to quit.
"""

import argparse
import time
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO

# BGR colours; each tracked object ID gets one colour
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Object detection and tracking")
    parser.add_argument("--source", default="0",
                        help="0 for webcam, or path to a video file")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLO model (yolov8n.pt is small and fast)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="minimum confidence (0 to 1)")
    parser.add_argument("--save", action="store_true",
                        help="save the result as output.mp4")
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.model)  # downloads the model on first run
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

    trails = defaultdict(lambda: deque(maxlen=30))  # recent centre points per ID
    seen_ids = set()
    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Detect + track (persist=True keeps IDs between frames)
        results = model.track(frame, persist=True, conf=args.conf,
                              tracker="bytetrack.yaml", verbose=False)
        boxes = results[0].boxes
        counts = defaultdict(int)

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            ids = (boxes.id.cpu().numpy().astype(int)
                   if boxes.id is not None else [None] * len(xyxy))

            for (x1, y1, x2, y2), cls, conf, track_id in zip(xyxy, classes, confs, ids):
                name = model.names[cls]
                counts[name] += 1
                color = PALETTE[(track_id if track_id is not None else cls) % len(PALETTE)]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = (f"{name} {conf:.2f}" if track_id is None
                         else f"ID {track_id} {name} {conf:.2f}")
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if track_id is not None:
                    seen_ids.add(track_id)
                    trails[track_id].append(((x1 + x2) // 2, (y1 + y2) // 2))
                    points = trails[track_id]
                    for i in range(1, len(points)):  # draw the movement path
                        cv2.line(frame, points[i - 1], points[i], color, 2)

        # FPS and summary overlay
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
