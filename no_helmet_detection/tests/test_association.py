from src.no_helmet_detection.geometry import (
    Detection,
    match_head_to_person,
    match_person_to_motorcycle,
    match_plate_to_motorcycle,
)


def test_head_and_person_associate_to_a_motorcycle() -> None:
    head = Detection((44, 5, 58, 20), "no_helmet", 0.9)
    person = Detection((30, 0, 72, 100), "person", 0.95)
    motorcycle = Detection((12, 72, 90, 125), "motorcycle", 0.9, 17)

    matched_person = match_head_to_person(head, [person])

    assert matched_person == person
    assert match_person_to_motorcycle(matched_person, [motorcycle]) == motorcycle


def test_plate_associates_by_its_position_inside_motorcycle_box() -> None:
    motorcycle = Detection((10, 30, 100, 120), "motorcycle", 0.9, 4)
    plate = Detection((68, 96, 91, 108), "license_plate", 0.85)

    assert match_plate_to_motorcycle(plate, [motorcycle]) == motorcycle


def test_plate_far_from_all_motorcycles_is_not_associated() -> None:
    motorcycle = Detection((10, 30, 100, 120), "motorcycle", 0.9, 4)
    plate = Detection((150, 150, 180, 165), "license_plate", 0.85)

    assert match_plate_to_motorcycle(plate, [motorcycle]) is None
