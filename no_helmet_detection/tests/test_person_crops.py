import numpy as np

from src.no_helmet_detection.geometry import Detection
from src.no_helmet_detection.pipeline import (
    _associate_no_helmet_heads,
    _detect_person_heads,
    _motorcycle_rider_candidates,
)


class StubDetector:
    def detect(self, crop: np.ndarray, confidence: float) -> list[Detection]:
        assert crop.shape[:2] == (250, 260)
        assert confidence == 0.25
        return [Detection((110, 30, 120, 50), "no-helmet", 0.9)]


def test_head_box_from_person_crop_is_restored_to_frame_coordinates() -> None:
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    person = Detection((100, 100, 140, 200), "person", 0.9, 2)

    heads = _detect_person_heads(StubDetector(), frame, [person], confidence=0.25)

    assert len(heads) == 1
    assert heads[0].box == (110, 105, 120, 125)
    assert heads[0].class_name == "no-helmet"


def test_no_helmet_person_is_only_a_candidate_without_a_motorcycle_match() -> None:
    head = Detection((40, 10, 55, 25), "no-helmet", 0.9)
    person = Detection((30, 10, 70, 100), "person", 0.9, 4)

    subjects, matched_people, unmatched_people, motorcycle_by_person = (
        _associate_no_helmet_heads([head], [person], [])
    )

    assert subjects == {}
    assert matched_people == {4: person}
    assert unmatched_people == {4}
    assert motorcycle_by_person == {}


def test_no_helmet_person_is_associated_with_a_matched_motorcycle() -> None:
    head = Detection((40, 10, 55, 25), "no-helmet", 0.9)
    person = Detection((30, 10, 70, 100), "person", 0.9, 4)
    motorcycle = Detection((20, 70, 90, 120), "motorcycle", 0.9, 8)

    subjects, _, unmatched_people, motorcycle_by_person = (
        _associate_no_helmet_heads([head], [person], [motorcycle])
    )

    assert subjects == {("motorcycle", 8): motorcycle}
    assert unmatched_people == set()
    assert motorcycle_by_person == {4: 8}


def test_head_inference_candidates_are_limited_to_motorcycle_riders() -> None:
    rider = Detection((30, 10, 70, 100), "person", 0.9, 4)
    pedestrian = Detection((150, 10, 190, 100), "person", 0.9, 5)
    motorcycle = Detection((20, 70, 90, 120), "motorcycle", 0.9, 8)

    candidates = _motorcycle_rider_candidates(
        [rider, pedestrian],
        [motorcycle],
    )

    assert candidates == [rider]
