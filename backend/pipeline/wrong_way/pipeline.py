# backend/pipeline/wrong_way/pipeline.py
# ---------------------------------------------------------
# connects lane association and wrong-way verification.

from .contracts import Lane, VehicleTrack, WrongWayEvent
from .lane_association import LaneAssociation
from .violation import WrongWayVerifier


class WrongWayPipeline:
    def __init__(
        self,
        persistence_frames: int = 10,
        min_displacement: float = 10.0,
        dot_threshold: float = -0.3,
    ):
        self.lane_association = LaneAssociation()

        self.verifier = WrongWayVerifier(
            persistence_frames=persistence_frames,
            min_displacement=min_displacement,
            dot_threshold=dot_threshold,
        )

    def process(
        self,
        tracks: list[VehicleTrack],
        lanes: list[Lane],
        frame_number: int,
    ) -> list[WrongWayEvent]:

        events = []

        for track in tracks:
            lane = self.lane_association.find_lane(
                track,
                lanes,
            )

            if lane is None:
                track.lane_id = None
                continue

            track.lane_id = lane.lane_id

            event = self.verifier.update(
                track=track,
                lane=lane,
                frame_number=frame_number,
            )

            if event is not None:
                events.append(event)

        return events

    def remove_track(self, track_id: int) -> None:
        self.verifier.remove(track_id)