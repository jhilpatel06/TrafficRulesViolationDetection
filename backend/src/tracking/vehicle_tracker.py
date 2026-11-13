# backend/src/tracking/vehicle_tracker.py
#--------------------------------------------------------------------------------
# assigns a persistent ID by matching the centroids of current detections with those from the previous frame.

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    track_id: int
    bbox: tuple
    centroid: tuple
    class_id: int
    class_name: str
    confidence: float
    disappeared: int = 0
    history: list = field(default_factory=list)


class VehicleTracker:
    def __init__(
        self,
        max_disappeared: int = 10,
        max_distance: float = 80.0,
    ):
        self.next_track_id = 0
        self.tracks = {}

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    @staticmethod
    def _get_centroid(bbox):
        x1, y1, x2, y2 = bbox

        return (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2),
        )

    def _register(self, detection):
        bbox = detection["bbox"]
        centroid = self._get_centroid(bbox)

        track = Track(
            track_id=self.next_track_id,
            bbox=bbox,
            centroid=centroid,
            class_id=detection["class_id"],
            class_name=detection["class_name"],
            confidence=detection["confidence"],
            history=[centroid],
        )

        self.tracks[self.next_track_id] = track
        self.next_track_id += 1

    def _deregister(self, track_id):
        del self.tracks[track_id]

    def update(self, detections):
        """
        Input: detections returned by VehicleDetector.detect()

        Output: currently active Track objects.
        """

        # No detections in this frame
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].disappeared += 1

                if (
                    self.tracks[track_id].disappeared
                    > self.max_disappeared
                ):
                    self._deregister(track_id)

            return list(self.tracks.values())

        # No existing tracks: register all detections
        if len(self.tracks) == 0:
            for detection in detections:
                self._register(detection)

            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())

        old_centroids = np.array([
            self.tracks[track_id].centroid
            for track_id in track_ids
        ])

        new_centroids = np.array([
            self._get_centroid(detection["bbox"])
            for detection in detections
        ])

        distances = cdist(old_centroids, new_centroids)

        rows, cols = linear_sum_assignment(distances)

        matched_tracks = set()
        matched_detections = set()

        # Match existing tracks with new detections
        for row, col in zip(rows, cols):
            if distances[row, col] > self.max_distance:
                continue

            track_id = track_ids[row]
            detection = detections[col]
            track = self.tracks[track_id]

            centroid = self._get_centroid(detection["bbox"])

            track.bbox = detection["bbox"]
            track.centroid = centroid
            track.class_id = detection["class_id"]
            track.class_name = detection["class_name"]
            track.confidence = detection["confidence"]
            track.disappeared = 0
            track.history.append(centroid)

            matched_tracks.add(track_id)
            matched_detections.add(col)

        # Unmatched existing tracks
        for track_id in track_ids:
            if track_id not in matched_tracks:
                self.tracks[track_id].disappeared += 1

                if (
                    self.tracks[track_id].disappeared
                    > self.max_disappeared
                ):
                    self._deregister(track_id)

        # Unmatched detections become new tracks
        for index, detection in enumerate(detections):
            if index not in matched_detections:
                self._register(detection)

        return list(self.tracks.values())