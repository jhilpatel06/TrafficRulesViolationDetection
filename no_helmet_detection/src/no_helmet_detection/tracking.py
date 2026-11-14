from __future__ import annotations

from dataclasses import replace
from math import hypot

from .geometry import Box, Detection, box_area, box_center, intersection_area


def _iou(first: Box, second: Box) -> float:
    intersection = intersection_area(first, second)
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union > 0 else 0.0


class MotorcycleTracker:
    def __init__(self, max_gap: int = 5) -> None:
        self.max_gap = max_gap
        self._next_id = 1
        self._tracks: dict[int, tuple[Box, int]] = {}

    def update(self, detections: list[Detection], frame_number: int) -> list[Detection]:
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if frame_number - track[1] <= self.max_gap + 1
        }

        possible_matches: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            center = box_center(detection.box)
            for track_id, (previous_box, _) in self._tracks.items():
                previous_center = box_center(previous_box)
                scale = max(
                    detection.box[2] - detection.box[0],
                    detection.box[3] - detection.box[1],
                    previous_box[2] - previous_box[0],
                    previous_box[3] - previous_box[1],
                    1.0,
                )
                normalized_distance = hypot(
                    center[0] - previous_center[0],
                    center[1] - previous_center[1],
                ) / scale
                overlap = _iou(detection.box, previous_box)
                if overlap >= 0.05 or normalized_distance <= 2.5:
                    cost = 1.0 - overlap + normalized_distance * 0.15
                    possible_matches.append((cost, detection_index, track_id))

        matches_by_detection: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _, detection_index, track_id in sorted(possible_matches):
            if detection_index in matches_by_detection or track_id in used_tracks:
                continue
            matches_by_detection[detection_index] = track_id
            used_tracks.add(track_id)

        tracked: list[Detection] = []
        for detection_index, detection in enumerate(detections):
            track_id = matches_by_detection.get(detection_index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
            self._tracks[track_id] = (detection.box, frame_number)
            tracked.append(replace(detection, track_id=track_id))
        return tracked
