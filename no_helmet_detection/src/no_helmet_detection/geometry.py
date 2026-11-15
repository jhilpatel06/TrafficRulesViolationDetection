from __future__ import annotations

from dataclasses import dataclass


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    box: Box
    class_name: str
    confidence: float
    track_id: int | None = None


def box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(first: Box, second: Box) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def box_center(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def contains(box: Box, point: tuple[float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def expand_box(box: Box, horizontal: float, vertical: float) -> Box:
    width = box[2] - box[0]
    height = box[3] - box[1]
    return (
        box[0] - width * horizontal,
        box[1] - height * vertical,
        box[2] + width * horizontal,
        box[3] + height * vertical,
    )


def match_head_to_person(head: Detection, people: list[Detection]) -> Detection | None:
    center = box_center(head.box)
    head_area = max(box_area(head.box), 1.0)
    matches: list[tuple[float, Detection]] = []
    for person in people:
        coverage = intersection_area(head.box, person.box) / head_area
        if contains(person.box, center) or coverage >= 0.25:
            matches.append((coverage + (1.0 if contains(person.box, center) else 0.0), person))
    return max(matches, key=lambda match: match[0])[1] if matches else None


def match_person_to_motorcycle(
    person: Detection, motorcycles: list[Detection]
) -> Detection | None:
    person_area = max(box_area(person.box), 1.0)
    foot_point = ((person.box[0] + person.box[2]) / 2, person.box[3])
    matches: list[tuple[float, Detection]] = []
    for motorcycle in motorcycles:
        overlap = intersection_area(person.box, motorcycle.box) / person_area
        expanded_motorcycle = expand_box(motorcycle.box, horizontal=0.2, vertical=0.25)
        foot_inside = contains(expanded_motorcycle, foot_point)
        if overlap >= 0.02 or foot_inside:
            matches.append((overlap + (0.25 if foot_inside else 0.0), motorcycle))
    return max(matches, key=lambda match: match[0])[1] if matches else None


def match_plate_to_motorcycle(
    plate: Detection, motorcycles: list[Detection]
) -> Detection | None:
    center = box_center(plate.box)
    matches: list[tuple[float, Detection]] = []
    for motorcycle in motorcycles:
        expanded_motorcycle = expand_box(motorcycle.box, horizontal=0.15, vertical=0.2)
        if contains(expanded_motorcycle, center):
            in_original_box = contains(motorcycle.box, center)
            overlap = intersection_area(plate.box, motorcycle.box) / max(
                box_area(plate.box), 1.0
            )
            matches.append((overlap + (1.0 if in_original_box else 0.0), motorcycle))
    return max(matches, key=lambda match: match[0])[1] if matches else None


