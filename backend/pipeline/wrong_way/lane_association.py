# backend/pipeline/wrong_way/lane_association.py
# ---------------------------------------------------------
# Uses the bottom-centre of the vehicle bounding box.
# Checks which lane polygon contains that point.
# Returns the matching lane, or None if no lane contains it.

from .contracts import Lane, Point, VehicleTrack


class LaneAssociation:
    def __init__(self, transition_frames: int = 3):
        self.transition_frames = transition_frames

        self.assignments: dict[int, str] = {}
        self.candidates: dict[int, str] = {}
        self.candidate_counts: dict[int, int] = {}

    @staticmethod
    def get_vehicle_reference_point(
        bbox: tuple[float, float, float, float]
    ) -> Point:
        x1, y1, x2, y2 = bbox

        return Point(
            x=(x1 + x2) / 2,
            y=y2,
        )

    @staticmethod
    def is_point_inside_polygon(
        point: Point,
        polygon: list[Point],
    ) -> bool:
        inside = False
        j = len(polygon) - 1

        for i in range(len(polygon)):
            xi, yi = polygon[i].x, polygon[i].y
            xj, yj = polygon[j].x, polygon[j].y

            intersects = (
                (yi > point.y) != (yj > point.y)
                and point.x
                < (xj - xi) * (point.y - yi)
                / ((yj - yi) or 1e-10)
                + xi
            )

            if intersects:
                inside = not inside

            j = i

        return inside

    def find_lane(
        self,
        track: VehicleTrack,
        lanes: list[Lane],
    ) -> Lane | None:

        point = self.get_vehicle_reference_point(track.bbox)

        candidate = next(
            (
                lane for lane in lanes
                if self.is_point_inside_polygon(point, lane.polygon)
            ),
            None,
        )

        track_id = track.track_id

        if candidate is None:
            self.candidates.pop(track_id, None)
            self.candidate_counts.pop(track_id, None)
            return None

        current_lane_id = self.assignments.get(track_id)

        # Initial lane assignment
        if current_lane_id is None:
            self.assignments[track_id] = candidate.lane_id
            track.lane_id = candidate.lane_id
            return candidate

        # Vehicle remains in its current lane
        if candidate.lane_id == current_lane_id:
            self.candidates.pop(track_id, None)
            self.candidate_counts.pop(track_id, None)
            track.lane_id = current_lane_id
            return candidate

        # Vehicle appears to be changing lanes
        if self.candidates.get(track_id) != candidate.lane_id:
            self.candidates[track_id] = candidate.lane_id
            self.candidate_counts[track_id] = 1
        else:
            self.candidate_counts[track_id] += 1

        # Confirm the transition
        if self.candidate_counts[track_id] >= self.transition_frames:
            self.assignments[track_id] = candidate.lane_id
            self.candidates.pop(track_id, None)
            self.candidate_counts.pop(track_id, None)

            track.lane_id = candidate.lane_id
            return candidate

        # Do not evaluate wrong-way movement during a pending transition
        return None

    def remove(self, track_id: int) -> None:
        self.assignments.pop(track_id, None)
        self.candidates.pop(track_id, None)
        self.candidate_counts.pop(track_id, None)