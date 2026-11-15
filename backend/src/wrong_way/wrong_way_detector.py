# backend/src/wrong_way/wrong_way_detector.py

"""
Wrong-way vehicle detection based on centroid trajectory and lane direction.

Parameter coupling (IMPORTANT — read before tuning):

    history_window >= 4 is required. The smoothing step averages the first
    and last `max(2, history_window // 3)` centroids of the window; below
    4 the two slices overlap and movement collapses toward zero. Values of
    6+ give good noise rejection in practice.

    smooth_window is derived as max(2, history_window // 3), so the
    effective time span of the movement measurement is
    (history_window - smooth_window) frames. Changing history_window
    changes this span, which in turn changes what "moving" means.

    min_displacement is interpreted as velocity in pixels per frame.
    Recommended range: 1-3 px/frame for typical 1080p CCTV. If set too
    high, slowly-creeping wrong-way vehicles are treated as stationary.
    If set too low, tracker jitter (box wiggle on a parked car) counts as
    movement.

    direction_threshold must be negative. It is compared against
    cos(angle between vehicle motion and permitted lane direction):
        0.0  -> flags perpendicular movement (normal turns)
        -0.5 -> flags >120 degree deviations (recommended)
        -0.7 -> very strict, flags only near-opposite motion
"""

import numpy as np

from src.wrong_way.lane_regions import LaneRegions
from src.wrong_way.visualize_debug import annotate_frame
from src.helper import load_config


def _bbox_bottom_center(bbox):
    """Convert (x1, y1, x2, y2) to a (cx, y2) bottom-center tuple."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


class WrongWayDetector:
    # A track ID absent for more than this many update() calls is purged
    # entirely (history + counters + lane cache). 60 frames ≈ 2 s at 30 fps,
    # which comfortably exceeds ByteTrack's lost_track_buffer of 30.
    _TRACK_GRACE_FRAMES = 60

    # A cached _last_lane entry older than this many frames is treated
    # as stale and no longer used for direction evaluation.
    _LAST_LANE_MAX_AGE = 10

    def __init__(
        self,
        history_window: int,
        min_displacement: float,
        direction_threshold: float,
        required_consecutive_frames: int,
    ):
        # --- validation -------------------------------------------------
        if history_window < 4:
            raise ValueError(
                f"history_window must be >= 4 (got {history_window}). "
                "Smaller values make the smoothing slices overlap and "
                "silently disable detection."
            )
        if min_displacement <= 0:
            raise ValueError(
                f"min_displacement must be > 0 (got {min_displacement}). "
                "It is interpreted as velocity in pixels per frame."
            )
        if direction_threshold >= 0:
            raise ValueError(
                f"direction_threshold must be < 0 (got {direction_threshold}). "
                "0 flags perpendicular movement; positive values flag "
                "normal turns. Use -0.5 for typical wrong-way detection."
            )

        direction_config = load_config()["lane_directions"]
        self.lanes_cfg = direction_config["lanes"]
        self.lane_regions = LaneRegions(direction_config["lanes"])

        self.history_window = history_window
        self.min_displacement = float(min_displacement)
        self.direction_threshold = float(direction_threshold)
        self.required_consecutive_frames = required_consecutive_frames

        self.violation_counts = {}
        self.wrong_way_ids = set()

        # track_id -> (lane_dict, frame_number_when_stored)
        self._last_lane = {}

        # Per-track centroid history. Public so debug tools can read it.
        self.histories = {}

        # track_id -> consecutive missed frames (for grace-period cleanup)
        self._missed = {}

        # Monotonic frame counter, incremented once per update() call.
        self._frame_counter = 0

    # ------------------------------------------------------------------ helpers

    def _cleanup_stale_tracks(self, active_ids):
        """Age out state for track IDs absent from the current frame.

        Uses a grace period so brief occlusions (a car passing behind a
        pole, ByteTrack momentarily dropping a detection) do not wipe the
        trajectory history that makes direction detection possible.
        """
        # Active IDs: clear any miss counter
        for tid in active_ids:
            self._missed.pop(tid, None)

        # Known but inactive IDs: bump miss counter
        known = set(self.histories) | set(self.violation_counts) | set(self._last_lane)
        for tid in known:
            if tid not in active_ids:
                self._missed[tid] = self._missed.get(tid, 0) + 1

        # Purge anything past the grace period
        for tid in list(self._missed):
            if self._missed[tid] > self._TRACK_GRACE_FRAMES:
                self.histories.pop(tid, None)
                self.violation_counts.pop(tid, None)
                self._last_lane.pop(tid, None)
                self._missed.pop(tid, None)

    def _append_history(self, track_id, centroid):
        """Push a centroid onto the track's history and bound its length."""
        hist = self.histories.get(track_id)
        if hist is None:
            hist = []
            self.histories[track_id] = hist

        hist.append(centroid)

        # Keep at most 2×history_window so long-lived tracks don't grow
        # unbounded, while still giving us a full window to slice.
        max_len = self.history_window * 2
        if len(hist) > max_len:
            del hist[: len(hist) - max_len]

    def _resolve_lane(self, track_id, bbox):
        """Return the lane dict for this track, or None if unavailable.

        Uses a time-bounded fallback to the last known lane so a brief
        detector dropout inside a lane doesn't reset direction eval.
        Once the cached lane is older than _LAST_LANE_MAX_AGE frames,
        the fallback stops and the track is treated as out-of-lane.
        """
        lane = self.lane_regions.get_lane(bbox)

        if lane is not None:
            self._last_lane[track_id] = (lane, self._frame_counter)
            return lane

        cached = self._last_lane.get(track_id)
        if cached is None:
            return None

        cached_lane, cached_frame = cached
        if self._frame_counter - cached_frame > self._LAST_LANE_MAX_AGE:
            return None
        return cached_lane

    def _smooth_movement(self, history):
        if len(history) < self.history_window:
            return None

        recent = np.array(history[-self.history_window:], dtype=float)
        
        # Separate X and Y coordinates
        x = recent[:, 0]
        y = recent[:, 1]
        
        # Create a time array [0, 1, 2, ... N]
        t = np.arange(len(recent))
        
        # Fit a line using polynomial regression of degree 1 (a straight line)
        # This finds the true overall direction, ignoring wiggles
        poly_x = np.polyfit(t, x, 1)
        poly_y = np.polyfit(t, y, 1)
        
        # The slope (index 0) is the velocity per frame in X and Y
        vx = poly_x[0]
        vy = poly_y[0]
        
        movement = np.array([vx, vy])
        velocity = np.linalg.norm(movement)
        
        return movement, velocity

    def update(self, tracks, frame_skip: int = 1):
        self._frame_counter += 1

        active_ids = {t.track_id for t in tracks}
        self._cleanup_stale_tracks(active_ids)

        current_wrong_way_ids = set()

        for track in tracks:
            track_id = track.track_id
            bbox = track.bbox

            self._append_history(track_id, _bbox_bottom_center(bbox))

            lane = self._resolve_lane(track_id, bbox)
            if lane is None:
                # Not in any lane and no recent cached lane to fall back to.
                self.violation_counts[track_id] = 0
                continue

            sm = self._smooth_movement(self.histories.get(track_id, []))
            if sm is None:
                # Not enough history yet.
                continue

            movement, velocity = sm

            # Dividing by frame_skip gives us the true pixels/frame velocity.
            velocity = velocity / frame_skip

            # --- Stationary Handling ---
            if velocity < self.min_displacement:
                self.violation_counts[track_id] = 0
                continue

            movement = movement / np.linalg.norm(movement)
            permitted = np.array(lane["direction"], dtype=float)
            permitted = permitted / np.linalg.norm(permitted)

            # alignment = cos(angle between motion and permitted direction)
            #   +1.0 = perfectly aligned
            #    0.0 = perpendicular
            #   -1.0 = perfectly opposite (classic wrong-way)
            alignment = float(np.dot(movement, permitted))

            if alignment < self.direction_threshold:
                self.violation_counts[track_id] = (
                    self.violation_counts.get(track_id, 0) + 1
                )
            else:
                self.violation_counts[track_id] = 0

            if self.violation_counts.get(track_id, 0) >= self.required_consecutive_frames:
                current_wrong_way_ids.add(track_id)

        self.wrong_way_ids = current_wrong_way_ids
        return current_wrong_way_ids

    # ------------------------------------------------------------------- debug

    def debug_visualize(self, frame, tracks, frame_idx=0, csv_rows=None):
        """Diagnostic overlay — same public signature as before."""
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