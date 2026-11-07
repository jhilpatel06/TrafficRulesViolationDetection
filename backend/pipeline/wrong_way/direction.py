# backend/pipeline/wrong_way/direction.py
# -------------------------------------------------------------
# Uses the vehicle's trajectory to estimate movement direction.
# Ignores movement below the minimum displacement threshold.
# Normalizes the movement and lane direction vectors.
# Uses their dot product to identify opposite movement.

import math

from .contracts import Point, VehicleTrack


class DirectionEstimator:
    def __init__(
        self,
        min_displacement: float = 10.0,
        dot_threshold: float = -0.3,
    ):
        self.min_displacement = min_displacement
        self.dot_threshold = dot_threshold

    def estimate_direction(
        self,
        track: VehicleTrack,
    ) -> Point | None:
        points = track.smoothed_trajectory or track.trajectory

        if len(points) < 2:
            return None

        start = points[0]
        end = points[-1]

        dx = end.x - start.x
        dy = end.y - start.y

        distance = math.sqrt(dx**2 + dy**2)

        if distance < self.min_displacement:
            return None

        return Point(
            x=dx / distance,
            y=dy / distance,
        )

    def is_wrong_way(
        self,
        track: VehicleTrack,
        expected_direction: Point,
    ) -> bool:
        movement = self.estimate_direction(track)

        if movement is None:
            return False

        lane_magnitude = math.sqrt(
            expected_direction.x**2 + expected_direction.y**2
        )

        if lane_magnitude == 0:
            return False

        lane_x = expected_direction.x / lane_magnitude
        lane_y = expected_direction.y / lane_magnitude

        dot_product = (
            movement.x * lane_x
            + movement.y * lane_y
        )

        return dot_product < self.dot_threshold