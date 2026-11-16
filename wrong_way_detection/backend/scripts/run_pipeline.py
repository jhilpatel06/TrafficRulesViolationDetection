# backend/src/wrong_way/run_pipeline.py
#-------------------------------------------------------------------------------

import cv2
import csv
import json
import numpy as np
import re
from collections import Counter
from pathlib import Path
from datetime import datetime
import shutil

from src.vehicle_detector import VehicleDetector
from src.vehicle_tracker import VehicleTracker
from src.wrong_way.wrong_way_detector import WrongWayDetector
from src.helper import load_config


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
        "track_activation_threshold": tracker_config.get(
            "track_activation_threshold", 0.25
        ),
        "lost_track_buffer": tracker_config.get("lost_track_buffer", 30),
        "minimum_matching_threshold": tracker_config.get(
            "minimum_matching_threshold", 0.8
        ),
        "frame_rate": tracker_config.get("frame_rate", 30),
    }

    fieldnames = list(record.keys())
    file_exists = CSV_PATH.exists()

    with CSV_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists or CSV_PATH.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(record)


def get_experiment_dir(config, timestamp):
    """Creates and returns a dedicated directory for this experiment run."""
    experiment_name = config.get("experiment_name", "").strip()
    if not experiment_name:
        experiment_name = "experiment"

    safe_name = re.sub(r'[<>:"/\\|?*]', "_", experiment_name).strip(" .")
    if not safe_name:
        safe_name = "experiment"

    # Create folder like: experiments/241116_1700_w_upd_wwD_n05/
    exp_dir = EXPERIMENT_DIR / f"{timestamp}_{safe_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def save_experiment_results(
    config,
    video,
    total_frames,
    fps,
    class_counts,
    tracked_ids,
    wrong_way_ids,
    timestamp,
    save_files=True,
):
    experiment_name = config.get("experiment_name", "")
    results = {
        "experiment_name": experiment_name,
        "video_info": {
            "video_path": str(video),
            "total_frames": total_frames,
            "fps": fps,
        },
        "object_detection": {
            "total_unique_vehicles": sum(class_counts.values()),
            "class_frequencies": dict(class_counts),
        },
        "object_tracking": {
            "total_unique_vehicles": len(tracked_ids),
            "track_ids": sorted(tracked_ids),
        },
        "wrong_way_detection": {
            "total_violations": len(wrong_way_ids),
            "violating_track_ids": sorted(wrong_way_ids),
        },
    }

    exp_dir = get_experiment_dir(config, timestamp)
    
    if save_files:
        output_path = exp_dir / "results.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)

    print("\n========== EXPERIMENT RESULTS ==========")

    print("\nObject Detection")
    print("Total detections :", sum(class_counts.values()))
    print("Class frequencies:", dict(class_counts))

    print("\nObject Tracking")
    print("Total unique vehicles:", len(tracked_ids))
    print("Track IDs:", sorted(tracked_ids))

    print("\nWrong-Way Detection")
    print("Total violating vehicles:", len(wrong_way_ids))
    print("Violating track IDs     :", sorted(wrong_way_ids))

    if save_files:
        print(f"\nAll results saved to folder: backend\experiments\{experiment_name}")
    else:
        print("\n[save_result=false] Files were not written to disk.")


def main():
    config = load_config()
    video = config["video_path"]

    save_result = bool(config.get("save_result", True))
    print(f"[INFO] save_result = {save_result}")

    timestamp = datetime.now().strftime("%y%m%d_%H%M")

    detector_config = config["detector_config"].copy()
    frame_skip = detector_config.pop("frame_skip", 1)
    if frame_skip < 1:
        frame_skip = 1

    # Initialization
    detector = VehicleDetector(weights="yolo26s.pt", **detector_config)
    tracker = VehicleTracker(**config["tracker_config"])
    wrong_way_detector = WrongWayDetector(**config["wrong_way_detector_config"])

    # Class filter derived from config's vehicle_class_ids
    tracked_classes = detector.allowed_class_names
    print(f"[INFO] tracked classes: {sorted(tracked_classes)}")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"total frames in video: {total_frames}")
    print(f"fps: {cap.get(cv2.CAP_PROP_FPS)}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 5 or fps > 240:
        print(f"[WARN] bad source fps ({fps!r}), defaulting to 30.0")
        fps = 30.0
    else:
        fps = round(fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("frame size:", width, "x", height)

    # --- Writers: main + debug, created only when saving ------------------
    output_video_path = None
    writer = None
    debug_video_path = None
    debug_csv_path = None
    debug_writer = None
    debug_csv_rows = []
    exp_dir = None

    if save_result:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        exp_dir = get_experiment_dir(config, timestamp)

        # Put output.mp4 directly in the folder
        output_video_path = exp_dir / "output.mp4"
        writer = cv2.VideoWriter(
            str(output_video_path), fourcc, fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"VideoWriter failed: {output_video_path}")

        # Put debug files in the same folder
        debug_video_path = exp_dir / "debug.mp4"
        debug_csv_path = exp_dir / "debug.csv"
        debug_writer = cv2.VideoWriter(
            str(debug_video_path), fourcc, fps, (width, height)
        )
        if not debug_writer.isOpened():
            raise RuntimeError(f"VideoWriter failed: {debug_video_path}")

    # --- State variables ---
    current_frame = 0
    completed_successfully = False
    track_classes = {}
    tracked_ids = set()
    wrong_way_ids = set()
        
    # Store the last processed tracks to draw on skipped frames
    last_valid_tracks = []

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                completed_successfully = True
                break

            current_frame += 1

            # --- INFERENCE BLOCK (Runs every N frames) ---
            if (current_frame - 1) % frame_skip == 0:
                detections = detector.detect(frame)
                tracks = tracker.update(detections)

                # Filter by class
                valid_tracks = [
                    t for t in tracks
                    if t.class_name in tracked_classes
                ]
                last_valid_tracks = valid_tracks # Save for skipped frames

                for track in valid_tracks:
                    tracked_ids.add(track.track_id)
                    if track.track_id not in track_classes:
                        track_classes[track.track_id] = track.class_name

                # Update wrong-way logic (Pass frame_skip to adjust velocity)
                current_wrong_way_ids = wrong_way_detector.update(valid_tracks, frame_skip=frame_skip)
                wrong_way_ids.update(current_wrong_way_ids)

                # Debug overlay
                if debug_writer is not None:
                    debug_frame = wrong_way_detector.debug_visualize(
                        frame, valid_tracks, frame_idx=current_frame, csv_rows=debug_csv_rows
                    )
                    for track in valid_tracks:
                        if hasattr(track, 'history') and len(track.history) > 1:
                            path_color = (139, 0, 0) # BGR for Dark Blue
                            points = np.array(track.history, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(debug_frame, [points], isClosed=False, color=path_color, thickness=2)
                            cv2.circle(debug_frame, (int(track.history[-1][0]), int(track.history[-1][1])), 3, path_color, -1)

                    debug_writer.write(debug_frame)

            # --- RENDER BLOCK (Runs every frame) ---
            # On skipped frames, this uses `last_valid_tracks` from the last processed frame.
            for track in last_valid_tracks:
                x1, y1, x2, y2 = map(int, track.bbox)
                is_in_lane = wrong_way_detector.lane_regions.get_lane(track.bbox) is not None

                if track.track_id in wrong_way_ids:
                    color = (0, 0, 255)
                    label = f"ID: {track.track_id} | WRONG WAY"
                elif is_in_lane:
                    color = (0, 255, 0)
                    label = f"ID: {track.track_id}"
                else:
                    color = (0, 255, 255)
                    label = f"ID: {track.track_id} | OUT OF LANE"

                if hasattr(track, 'history') and len(track.history) > 1:
                    # Convert history to a format OpenCV understands: [[[x, y]], [[x, y]]]
                    points = np.array(track.history, dtype=np.int32).reshape((-1, 1, 2))
                    
                    # Draw the line (thickness 2)
                    path_color = (255, 0, 0)
                    cv2.polylines(frame, [points], isClosed=False, color=path_color, thickness=2)
                    
                    # Draw a dot at the current position
                    cv2.circle(frame, (int(track.history[-1][0]), int(track.history[-1][1])), 3, color, -1)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


            cv2.imshow("Vehicle Detection and Tracking", frame)
            
            if writer is not None:
                # Write EVERY frame to keep the output video 30fps and smooth
                writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if debug_writer is not None:
            debug_writer.release()
        cv2.destroyAllWindows()

        if completed_successfully:
            # Debug CSV written only on a completed run
            if debug_writer is not None and debug_csv_rows:
                with debug_csv_path.open("w", newline="", encoding="utf-8") as f:
                    csv_writer = csv.DictWriter(
                        f, fieldnames=list(debug_csv_rows[0].keys())
                    )
                    csv_writer.writeheader()
                    csv_writer.writerows(debug_csv_rows)

            class_counts = Counter(track_classes.values())

            if save_result:
                save_experiment_config(config)

            save_experiment_results(
                config=config,
                video=video,
                total_frames=total_frames,
                fps=fps,
                class_counts=class_counts,
                tracked_ids=tracked_ids,
                wrong_way_ids=wrong_way_ids,
                timestamp=timestamp,
                save_files=save_result,
            )

        else:
            # Aborted (q or exception) — remove the entire partial folder
            if exp_dir is not None and exp_dir.exists():
                shutil.rmtree(exp_dir)
            print("\n[aborted] No files were saved. Folder deleted.")


if __name__ == "__main__":
    main()