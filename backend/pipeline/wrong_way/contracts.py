# backend/pipeline/wrong_way/contracts.py

from dataclasses import dataclass, field
from typing import Optional

# Stores a coordinate
@dataclass
class Point:
    x: float
    y: float

# Stores a vehicle's identity, bounding box, trajectory and lane assignment.
@dataclass
class VehicleTrack:
    track_id: int
    vehicle_class: str
    bbox: tuple[float, float, float, float]

    trajectory: list[Point] = field(default_factory=list)

    lane_id: Optional[str] = None
    status: str = "active"
    last_seen_frame: int = 0
    smoothed_trajectory: list[Point] = field(default_factory=list)

# Stores lane boundaries and expected travel direction.
@dataclass
class Lane:
    lane_id: str
    polygon: list[Point]
    direction_start: Point
    direction_end: Point

# Stores a confirmed wrong-way detection.
@dataclass
class WrongWayEvent:
    track_id: int
    lane_id: str
    frame_number: int
    confidence: float