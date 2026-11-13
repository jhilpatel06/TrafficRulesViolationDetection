# backend/src/wrong_way/wrong_way_detector.py

from networkx import config
import yaml
import numpy as np

from src.wrong_way.lane_regions import LaneRegions
from src.utils.helper import load_config


class WrongWayDetector:
    def __init__(
        self,
        history_window=8,
        min_displacement=25,
        direction_threshold=-0.5,
        required_consecutive_frames=3
    ):
        direction_config = load_config()["lane_directions"]
        self.lane_regions = LaneRegions(direction_config["lanes"])

        self.history_window = history_window
        self.min_displacement = min_displacement
        self.direction_threshold = direction_threshold
        self.required_consecutive_frames = required_consecutive_frames

        self.violation_counts = {}
        self.wrong_way_ids = set()

    def update(self, tracks):
        current_wrong_way_ids = set()

        for track in tracks:
            track_id = track.track_id

            lane = self.lane_regions.get_lane(track.bbox)

            if lane is None:
                self.violation_counts[track_id] = 0
                continue

            history = track.history

            if len(history) < 2:
                continue

            # Compare recent positions
            recent = history[-self.history_window:]

            start = np.array(recent[0], dtype=float)
            end = np.array(recent[-1], dtype=float)

            movement = end - start

            distance = np.linalg.norm(movement)

            if distance < self.min_displacement:
                continue

            movement = movement / distance

            permitted = np.array(
                lane["direction"],
                dtype=float
            )

            permitted = permitted / np.linalg.norm(permitted)

            # Negative value means movement opposes lane direction
            alignment = float(np.dot(movement, permitted))

            if alignment < self.direction_threshold:
                self.violation_counts[track_id] = (
                    self.violation_counts.get(track_id, 0) + 1
                )
            else:
                self.violation_counts[track_id] = 0

            if (
                self.violation_counts.get(track_id, 0)
                >= self.required_consecutive_frames
            ):
                current_wrong_way_ids.add(track_id)

        self.wrong_way_ids = current_wrong_way_ids

        return current_wrong_way_ids
