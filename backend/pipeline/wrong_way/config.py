from pathlib import Path

import yaml

from .contracts import Lane, Point


def load_wrong_way_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or not isinstance(
        config.get("wrong_way"), dict
    ):
        raise ValueError("Invalid wrong-way configuration")

    return config["wrong_way"]


def load_lanes(config: dict) -> list[Lane]:
    camera_config = config.get("camera", {})
    lane_configs = camera_config.get("lanes", [])

    if not isinstance(lane_configs, list):
        raise ValueError("Camera lanes must be a list")

    lanes = []
    lane_ids = set()

    for lane_config in lane_configs:
        lane_id = lane_config["lane_id"]

        if lane_id in lane_ids:
            raise ValueError(f"Duplicate lane ID: {lane_id}")

        lane_ids.add(lane_id)

        polygon = [
            Point(float(x), float(y))
            for x, y in lane_config["polygon"]
        ]

        if len(polygon) < 3:
            raise ValueError(
                f"Lane {lane_id} must have at least 3 polygon points"
            )

        direction = lane_config["direction"]

        start = Point(*map(float, direction["start"]))
        end = Point(*map(float, direction["end"]))

        if start == end:
            raise ValueError(
                f"Lane {lane_id} has a zero-length direction vector"
            )

        lanes.append(
            Lane(
                lane_id=lane_id,
                polygon=polygon,
                direction_start=start,
                direction_end=end,
            )
        )

    return lanes