# backend/tests/wrong_way/test_config.py
from pathlib import Path

import pytest

from backend.pipeline.wrong_way.config import (
    load_wrong_way_config,
    load_lanes,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "wrong_way.yaml"
)


def test_load_wrong_way_config():
    config = load_wrong_way_config(CONFIG_PATH)

    assert config["enabled"] is True
    assert "camera" in config


def test_load_lanes():
    config = load_wrong_way_config(CONFIG_PATH)

    lanes = load_lanes(config)

    assert len(lanes) == 1
    assert lanes[0].lane_id == "lane_1"
    assert len(lanes[0].polygon) == 4


def test_duplicate_lane_ids_rejected():
    config = {
        "camera": {
            "lanes": [
                {
                    "lane_id": "lane_1",
                    "polygon": [[0, 0], [10, 0], [10, 10]],
                    "direction": {
                        "start": [0, 0],
                        "end": [10, 10],
                    },
                },
                {
                    "lane_id": "lane_1",
                    "polygon": [[20, 0], [30, 0], [30, 10]],
                    "direction": {
                        "start": [20, 0],
                        "end": [30, 10],
                    },
                },
            ]
        }
    }

    with pytest.raises(ValueError, match="Duplicate lane ID"):
        load_lanes(config)


def test_invalid_polygon_rejected():
    config = {
        "camera": {
            "lanes": [
                {
                    "lane_id": "lane_1",
                    "polygon": [[0, 0], [10, 0]],
                    "direction": {
                        "start": [0, 0],
                        "end": [10, 10],
                    },
                }
            ]
        }
    }

    with pytest.raises(ValueError, match="at least 3 polygon points"):
        load_lanes(config)