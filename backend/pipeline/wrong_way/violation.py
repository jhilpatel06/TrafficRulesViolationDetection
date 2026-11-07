# backend/pipeline/wrong_way/violation.py
# ----------------------------------------------------------------------------
# Counts consecutive frames in which a vehicle moves against its assigned lane.
# Confirms a violation only after the configured persistence threshold.
# Emits only one event per track.
# Resets the counter when wrong-way movement is no longer detected.

from .contracts import Lane, WrongWayEvent, VehicleTrack
from .direction import DirectionEstimator


class WrongWayVerifier:
    def __init__(
        self,
        persistence_frames: int = 10,
        min_displacement: float = 10.0,
        dot_threshold: float = -0.3,
    ):
        self.persistence_frames = persistence_frames

        self.direction_estimator = DirectionEstimator(
            min_displacement=min_displacement,
            dot_threshold=dot_threshold,
        )

        self.violation_counts: dict[int, int] = {}
        self.confirmed_tracks: set[int] = set()

    def update(
        self,
        track: VehicleTrack,
        lane: Lane,
        frame_number: int,
    ) -> WrongWayEvent | None:

        if track.track_id in self.confirmed_tracks:
            return None

        expected_direction = type(lane.direction_start)(
            x=lane.direction_end.x - lane.direction_start.x,
            y=lane.direction_end.y - lane.direction_start.y,
        )

        is_wrong_way = self.direction_estimator.is_wrong_way(
            track,
            expected_direction,
        )

        if not is_wrong_way:
            self.violation_counts[track.track_id] = 0
            return None

        self.violation_counts[track.track_id] = (
            self.violation_counts.get(track.track_id, 0) + 1
        )

        if (
            self.violation_counts[track.track_id]
            < self.persistence_frames
        ):
            return None

        self.confirmed_tracks.add(track.track_id)

        return WrongWayEvent(
            track_id=track.track_id,
            lane_id=lane.lane_id,
            frame_number=frame_number,
            confidence=0.85,
        )

    def remove(self, track_id: int) -> None:
        self.violation_counts.pop(track_id, None)
        self.confirmed_tracks.discard(track_id)