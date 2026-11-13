# backend/src/wrong_way/lane_regions.py
"""Load CVAT polygon tracks and assign tracked vehicle points to road regions."""
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np


class LaneRegions:
    def __init__(self, annotation_xml: str | Path, frame_width: int, frame_height: int):
        self.annotation_xml = Path(annotation_xml)
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.polygons = self._load_polygons()

    @staticmethod
    def _parse_points(points_text: str) -> np.ndarray:
        points = []
        for pair in points_text.split(";"):
            x_str, y_str = pair.split(",")
            points.append((float(x_str), float(y_str)))
        return np.asarray(points, dtype=np.float32)

    def _load_polygons(self) -> dict[str, np.ndarray]:
        if not self.annotation_xml.exists():
            raise FileNotFoundError(f"CVAT XML not found: {self.annotation_xml}")

        root = ET.parse(self.annotation_xml).getroot()
        original_size = root.find("./meta/task/original_size")
        if original_size is None:
            raise ValueError("Could not find original_size in CVAT annotations.xml")

        source_w = float(original_size.findtext("width"))
        source_h = float(original_size.findtext("height"))
        if source_w <= 0 or source_h <= 0:
            raise ValueError("Invalid source video dimensions in annotations.xml")

        scale_x = self.frame_width / source_w
        scale_y = self.frame_height / source_h
        polygons = {}

        for track in root.findall("track"):
            label = track.attrib.get("label")
            # All exported frames in this project carry the same polygon.
            # Read the first visible polygon for each track.
            polygon_element = next(
                (p for p in track.findall("polygon") if p.attrib.get("outside", "0") == "0"),
                None,
            )
            if polygon_element is None:
                continue

            points = self._parse_points(polygon_element.attrib["points"])
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y
            polygons[label] = np.round(points).astype(np.int32).reshape((-1, 1, 2))

        required = {"up", "Down"}
        missing = required - polygons.keys()
        if missing:
            raise ValueError(f"Missing expected CVAT labels: {sorted(missing)}. Found: {list(polygons)}")
        return polygons

    def lane_at(self, point: tuple[float, float]) -> str | None:
        """Return 'up', 'Down', None (outside), or 'overlap' if in both polygons."""
        x, y = float(point[0]), float(point[1])
        matches = [
            label for label, polygon in self.polygons.items()
            if cv2.pointPolygonTest(polygon, (x, y), False) >= 0
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 2:
            return "overlap"
        return None

    def draw(self, frame: np.ndarray, alpha: float = 0.25) -> np.ndarray:
        """Optional debug overlay; OpenCV colors are BGR."""
        overlay = frame.copy()
        colors = {"up": (250, 183, 50), "Down": (61, 245, 61)}
        for label, polygon in self.polygons.items():
            cv2.fillPoly(overlay, [polygon], colors.get(label, (255, 255, 255)))
            cv2.polylines(frame, [polygon], True, colors.get(label, (255, 255, 255)), 2)
            x, y = polygon.reshape(-1, 2)[0]
            cv2.putText(frame, label, (int(x), max(25, int(y) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors.get(label, (255, 255, 255)), 2)
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def vehicle_bottom_center(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """Approximate road-contact point of a vehicle bounding box."""
    return ((x1 + x2) / 2.0, y2)


def is_wrong_way(track_points: list[tuple[float, float]], lane: str,
                 min_motion_px: float = 12.0, min_opposition: float = 0.35) -> bool:
    """Basic vertical-image-direction test.

    track_points must be chronological bottom-centre points (x, y).
    'up' means image-up (negative y); 'Down' means image-down (positive y).
    Tune/replace the expected vectors if legal traffic follows a diagonal image path.
    """
    if lane not in {"up", "Down"} or len(track_points) < 2:
        return False

    # Use a short history window to reduce noisy frame-to-frame movement.
    start = np.asarray(track_points[0], dtype=np.float32)
    end = np.asarray(track_points[-1], dtype=np.float32)
    displacement = end - start
    distance = float(np.linalg.norm(displacement))
    if distance < min_motion_px:
        return False  # insufficient movement to make a reliable decision

    actual = displacement / distance
    expected = np.array([0.0, -1.0] if lane == "up" else [0.0, 1.0], dtype=np.float32)
    alignment = float(np.dot(actual, expected))
    return alignment < -min_opposition
