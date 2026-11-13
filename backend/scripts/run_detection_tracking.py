# backend/src/wrong_way/run_detection_tracking.py
#-------------------------------------------------------------------------------

import argparse
import cv2

from src.detection.vehicle_detector import VehicleDetector
from src.tracking.vehicle_tracker import VehicleTracker


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        default="data/videos/traffic.mp4",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to your YOLOv5 .pt weights",
    )
    parser.add_argument(
        "--yolov5-repo",
        required=True,
        help="Path to the cloned YOLOv5 repository",
    )

    args = parser.parse_args()

    detector = VehicleDetector(
        weights=args.weights,
        yolov5_repo=args.yolov5_repo,
        confidence=0.25,
        vehicle_class_ids=None,  # Change if your model has non-vehicle classes
    )

    tracker = VehicleTracker(
        max_disappeared=10,
        max_distance=80,
    )

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Stage 1: Detect vehicles
        detections = detector.detect(frame)

        # Stage 2: Assign persistent IDs
        tracks = tracker.update(detections)

        # Stage 3: Visualize
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            cx, cy = track.centroid

            # Skip drawing tracks temporarily missing from the frame
            if track.disappeared > 0:
                continue

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = (
                f"ID: {track.track_id} "
                f"{track.class_name} "
                f"{track.confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            cv2.circle(
                frame,
                (cx, cy),
                4,
                (0, 0, 255),
                -1,
            )

        cv2.imshow("Vehicle Detection and Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()