# backend/src/tracking/vehicle_tracker.py

import numpy as np
import supervision as sv
from collections import deque
from trackers import ByteTrackTracker


class Track:
    """Lightweight container so run_pipeline stays unchanged."""
    # Added "history" to slots
    __slots__ = ("track_id", "bbox", "class_name", "confidence", "history")

    def __init__(self, track_id, bbox, class_name, confidence=1.0, history=None):
        self.track_id = track_id
        self.bbox = bbox                 # (x1, y1, x2, y2)
        self.class_name = class_name
        self.confidence = confidence
        # Store as a tuple so it's immutable once passed to the pipeline
        self.history = history if history is not None else ()


class VehicleTracker:
    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 0.5,
        track_activation_threshold: float = 0.25,
        minimum_consecutive_frames: int = 1,
        **kwargs,
    ):
        # ByteTrackTracker accepts sv.Detections via update()
        self.tracker = ByteTrackTracker(
            track_activation_threshold=track_activation_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )
        
        # NEW: Dictionary to store centroid histories per track ID
        self._histories = {}
        # Keep last 15 frames (~0.5 second at 30fps)
        self._max_history_len = 15

    @staticmethod
    def _to_sv_detections(detections):
        if not detections:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=np.array([d["bbox"] for d in detections], dtype=np.float32),
            confidence=np.array([d["confidence"] for d in detections], dtype=np.float32),
            class_id=np.array([d["class_id"] for d in detections], dtype=np.int32),
            data={"class_name": np.array([d["class_name"] for d in detections])},
        )

    def update(self, detections):
        sv_dets = self._to_sv_detections(detections)
        tracked = self.tracker.update(sv_dets)

        if tracked is None or len(tracked) == 0:
            return []

        results = []
        active_ids = set()

        for i in range(len(tracked)):
            track_id = int(tracked.tracker_id[i])
            if track_id < 0:            
                continue
                
            active_ids.add(track_id)
            x1, y1, x2, y2 = map(int, tracked.xyxy[i].tolist())
            
            # Calculate bottom-center instead of centroid
            bx = (x1 + x2) / 2.0
            by = y2
            
            if track_id not in self._histories:
                self._histories[track_id] = deque(maxlen=self._max_history_len)
            
            self._histories[track_id].append((bx, by))
            
            results.append(
                Track(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    class_name=str(tracked.data["class_name"][i]),
                    history=tuple(self._histories[track_id]) # NEW: Pass the trail
                )
            )
            
        # Optional cleanup: clear history for tracks that disappeared
        # (ByteTrack keeps lost tracks internally, but we only need trails for active ones on screen)
        for old_id in list(self._histories.keys()):
            if old_id not in active_ids:
                del self._histories[old_id]

        return results