# backend/src/wrong_way/wrong_way_detector.py

import numpy as np

from src.wrong_way.lane_regions import LaneRegions
from src.wrong_way.visualize_debug import annotate_frame
from src.utils.helper import load_config


class WrongWayDetector:
    def __init__(
        self,
        history_window=int,       
        min_displacement=int,     
        direction_threshold=int,
        required_consecutive_frames=int
    ):
        direction_config = load_config()["lane_directions"]
        self.lanes_cfg = direction_config["lanes"] 
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

            # Wait until we have enough history to make a smooth calculation
            if len(history) < self.history_window:
                continue

            recent = history[-self.history_window:]

            # --- Trajectory Smoothing ---
            # Average the first 3 frames and last 3 frames of the window.
            # This completely eliminates false positives from 1-frame box wiggles.
            smooth_window = max(2, self.history_window // 3)
            start = np.mean(np.array(recent[:smooth_window], dtype=float), axis=0)
            end = np.mean(np.array(recent[-smooth_window:], dtype=float), axis=0)

            movement = end - start
            distance = np.linalg.norm(movement)

            # --- Stationary Vehicle Handling ---
            if distance < self.min_displacement:
                # If they aren't moving enough, they can't be going the wrong way.
                # Reset the count so stopped cars don't randomly trigger alarms.
                self.violation_counts[track_id] = 0
                continue

            movement = movement / distance
            permitted = np.array(lane["direction"], dtype=float)
            permitted = permitted / np.linalg.norm(permitted)

            # Calculate alignment (1.0 = perfect match, -1.0 = perfect wrong way)
            alignment = float(np.dot(movement, permitted))

            if alignment < self.direction_threshold:
                self.violation_counts[track_id] = (
                    self.violation_counts.get(track_id, 0) + 1
                )
            else:
                self.violation_counts[track_id] = 0

            # Only flag if the violation persists for required consecutive frames
            if self.violation_counts.get(track_id, 0) >= self.required_consecutive_frames:
                current_wrong_way_ids.add(track_id)

        self.wrong_way_ids = current_wrong_way_ids
        return current_wrong_way_ids

    def debug_visualize(self, frame, tracks, frame_idx=0, csv_rows=None):
        """Return a copy of `frame` annotated with lane polygons, centroids,
        bottom-center anchors, and per-track lane membership.

        Purely diagnostic — does not touch `violation_counts` or `wrong_way_ids`.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame read from the video.
        tracks : list
            The same track list that was passed to `update(tracks)`.
        frame_idx : int
            Frame counter (used in the debug CSV).
        csv_rows : list | None
            If provided, per-track rows are appended to it. Pass the same list
            across all frames, then write it to disk at the end.
        """
        if csv_rows is None:
            csv_rows = []
        return annotate_frame(
            frame,
            tracks,
            self.lane_regions,
            self.lanes_cfg,
            frame_idx,
            csv_rows,
        )