# backend/src/wrong_way/run_detection_tracking.py
#-------------------------------------------------------------------------------


import argparse
import cv2
from pathlib import Path

from src.detection.vehicle_detector import VehicleDetector
from src.tracking.vehicle_tracker import VehicleTracker
from src.wrong_way.wrong_way_detector import WrongWayDetector
from src.utils.helper import load_config


# Project paths
ROOT = Path(__file__).resolve().parents[1]
yolov5_repo = ROOT / "yolov5"
weights = yolov5_repo / "my_coco.pt"
video = load_config()["video_path"]


def main():

    # Load configuration once
    config = load_config()

    # Initialize YOLO model once
    detector = VehicleDetector(
        weights=weights,
        yolov5_repo=yolov5_repo,
        vehicle_class_ids=None,
        **config["detector_config"]
    )

    # Initialize tracker and wrong-way detector once
    tracker = VehicleTracker(**config["tracker_config"])

    wrong_way_detector = WrongWayDetector(
        **config["wrong_way_detector_config"]
    )

    # Open video
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Stage 1: Detect vehicles using the already-loaded model
            detections = detector.detect(frame)

            # Stage 2: Assign persistent IDs
            tracks = tracker.update(detections)
            wrong_way_ids = wrong_way_detector.update(tracks)

            # Stage 3: Visualize
            for track in tracks:
                x1, y1, x2, y2 = map(int, track.bbox)

                if track.track_id in wrong_way_ids:
                    color = (0, 0, 255)
                    status = "WRONG WAY"
                else:
                    color = (0, 255, 0)
                    status = "NORMAL"

                label = f"ID: {track.track_id} | {status}"

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), color, 2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            cv2.imshow("Vehicle Detection and Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
