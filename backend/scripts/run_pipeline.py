# backend/src/wrong_way/run_pipeline.py
#-------------------------------------------------------------------------------

import cv2
import csv
import json
import re
from collections import Counter
from pathlib import Path
from datetime import datetime

from src.detection.vehicle_detector import VehicleDetector
from src.tracking.vehicle_tracker import VehicleTracker
from src.wrong_way.wrong_way_detector import WrongWayDetector
from src.utils.helper import load_config


# Project paths
ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_DIR = ROOT / "experiments"
CSV_PATH = EXPERIMENT_DIR / "_experiment_configs.csv"

def save_experiment_config(config):
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    lane_directions = {
        lane["id"]: lane["direction"]
        for lane in config["lane_directions"]["lanes"]
    }

    wrong_way = config["wrong_way_detector_config"]
    
    detector_config = config["detector_config"]
    tracker_config = config["tracker_config"]

    record = {
        "exp_name": config.get("experiment_name", ""),
        "lane_right": json.dumps(lane_directions.get("right", [])),
        "lane_left": json.dumps(lane_directions.get("left", [])),
        "hist_window": wrong_way["history_window"],
        "min_disp": wrong_way["min_displacement"],
        "dir_thresh": wrong_way["direction_threshold"],
        "req_frames": wrong_way["required_consecutive_frames"],
        "conf": detector_config["confidence"],
        "iou": detector_config["iou"],
        "max_disappeared": tracker_config["max_disappeared"],
        "max_distance": tracker_config["max_distance"],
    }

    fieldnames = [
        "exp_name",
        "lane_right",
        "lane_left",
        "hist_window",
        "min_disp",
        "dir_thresh",
        "req_frames",
        "conf",
        "iou",
        "max_disappeared",
        "max_distance",
    ]

    file_exists = CSV_PATH.exists()

    with CSV_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists or CSV_PATH.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(record)

def get_experiment_video_path(config, timestamp):
    experiment_name = config.get("experiment_name", "").strip()

    if not experiment_name:
        experiment_name = "experiment"

    safe_name = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        experiment_name
    ).strip(" .")

    if not safe_name:
        safe_name = "experiment"

    filename = f"{timestamp}_{safe_name}.mp4"

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    return EXPERIMENT_DIR / filename


def save_experiment_results(
    config,
    video,
    total_frames,
    fps,
    class_counts,
    tracked_ids,
    wrong_way_ids,
    timestamp
):

    experiment_name = config.get("experiment_name", "")
    results = {
        "experiment_name": experiment_name,

        "video_info": {
            "video_path": str(video),
            "total_frames": total_frames,
            "fps": fps
        },

        "object_detection": {
            "total_unique_vehicles": sum(class_counts.values()),
            "class_frequencies": dict(class_counts)
        },

        "object_tracking": {
            "total_unique_vehicles": len(tracked_ids),
            "track_ids": sorted(tracked_ids)
        },

        "wrong_way_detection": {
            "total_violations": len(wrong_way_ids),
            "violating_track_ids": sorted(wrong_way_ids)
        }
    }

    output_path = get_experiment_results_path(config, timestamp)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    # Print results in terminal
    print("\n========== EXPERIMENT RESULTS ==========")

    print("\nObject Detection")
    print("Total detections :", sum(class_counts.values()))
    print("Class frequencies:", dict(class_counts))

    print("\nObject Tracking")
    print("Total unique vehicles:", sum(class_counts.values()))
    print("Track IDs:", sorted(tracked_ids))

    print("\nWrong-Way Detection")
    print("Total violating vehicles:", len(wrong_way_ids))
    print("Violating track IDs     :", sorted(wrong_way_ids))

    print(f"\nResults saved to     : backend\experiments\{timestamp}_{experiment_name}_results.json")
    print(f"Output video saved to: backend\experiments\{timestamp}_{experiment_name}.mp4")


def get_experiment_results_path(config, timestamp):

    experiment_name = config.get("experiment_name", "").strip()

    if not experiment_name:
        experiment_name = "experiment"

    safe_name = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        experiment_name
    ).strip(" .")

    if not safe_name:
        safe_name = "experiment"

    filename = f"{timestamp}_{safe_name}_results.json"

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    return EXPERIMENT_DIR / filename

def main():

    video = load_config()["video_path"]
    # Load configuration once
    config = load_config()
    timestamp = datetime.now().strftime("%y%m%d_%H%M")

    # Initialize YOLO model once
    detector = VehicleDetector(
        weights="yolo26n.pt",
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

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"total frames in video: {total_frames}")
    print(f"fps: {cap.get(cv2.CAP_PROP_FPS)}")

    output_video_path = get_experiment_video_path(config, timestamp)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 5 or fps > 240:
        print(f"[WARN] bad source fps ({fps!r}), defaulting to 30.0")
        fps = 30.0
    else:
        fps = round(fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"VideoWriter failed: {output_video_path}  "
            f"fourcc={fourcc} fps={fps} size=({width},{height})"
        )

    print("frame size:", width, "x", height)

    # debug_video_path = output_video_path.with_name(
    #     output_video_path.stem + "_debug.mp4"
    # )
    # debug_csv_path = output_video_path.with_name(
    #     output_video_path.stem + "_debug.csv"
    # )
    # debug_writer = cv2.VideoWriter(
    #     str(debug_video_path),
    #     fourcc,
    #     fps,
    #     (width, height),
    # )
    # if not debug_writer.isOpened():
    #     raise RuntimeError(
    #         f"VideoWriter failed: {debug_video_path}  "
    #         f"fourcc={fourcc} fps={fps} size=({width},{height})"
    #     )
    
    # debug_csv_rows = []

    current_frame = 0
    completed_successfully = False
    # Object detection statistics
    track_classes = {}
    # Unique tracked vehicle IDs
    tracked_ids = set()
    # Unique vehicles detected moving in the wrong direction
    wrong_way_ids = set()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                completed_successfully = True  # Add this line
                break

            current_frame += 1  # Add this line

            # Stage 1: Detect vehicles using the already-loaded model
            detections = detector.detect(frame)

            # Stage 2: Assign persistent IDs
            tracks = tracker.update(detections)

            for track in tracks:
                tracked_ids.add(track.track_id)

                if track.track_id not in track_classes:
                    track_classes[track.track_id] = track.class_name

            # Get wrong-way IDs in the current frame
            current_wrong_way_ids = wrong_way_detector.update(tracks)

            # Maintain unique wrong-way vehicle IDs across the video
            wrong_way_ids.update(current_wrong_way_ids)

            # # --- debug overlay (change 4) ---
            # debug_frame = wrong_way_detector.debug_visualize(
            #     frame,
            #     tracks,
            #     frame_idx=current_frame,
            #     csv_rows=debug_csv_rows,
            # )
            # debug_writer.write(debug_frame)

            # Stage 3: Visualize
            for track in tracks:
                x1, y1, x2, y2 = map(int, track.bbox)

                if track.track_id in wrong_way_ids:
                    color = (0, 0, 255)
                    label = f"ID: {track.track_id} | WRONG WAY"
                else:
                    color = (0, 255, 0)
                    label = f"ID: {track.track_id}"

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
            writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        writer.release()
        # debug_writer.release()                  
        cv2.destroyAllWindows()

        # # NEW: dump the per-frame debug CSV
        # if debug_csv_rows:
        #     with debug_csv_path.open("w", newline="", encoding="utf-8") as f:
        #         csv_writer = csv.DictWriter(
        #             f, fieldnames=list(debug_csv_rows[0].keys())
        #         )
        #         csv_writer.writeheader()
        #         csv_writer.writerows(debug_csv_rows)

        # Replace your single save_experiment_config(config) line with this block:
        if completed_successfully:

            class_counts = Counter(track_classes.values())
            # Save existing experiment configuration to CSV
            save_experiment_config(config)

            # Save detection, tracking and violation results to JSON
            save_experiment_results(
                config=config,
                video=video,
                total_frames=total_frames,
                fps=fps,
                class_counts=class_counts,
                tracked_ids=tracked_ids,
                wrong_way_ids=wrong_way_ids,
                timestamp = timestamp
            )

        else:
            if output_video_path.exists():
                output_video_path.unlink()


if __name__ == "__main__":
    main()
