from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import cv2
import requests

from .events import EventLedger, ViolationEvent
from .geometry import (
    Detection,
    match_head_to_person,
    match_person_to_motorcycle,
    match_plate_to_motorcycle,
)
from .onnx_models import OnnxDetector, suppress_duplicates
from .tracking import ObjectTracker


HELMET_REPO = "Nqhia/helmet-detection"
HELMET_COMMIT = "0c14d153a8e44696a31507dd428d06517a9648c3"
HELMET_FILENAME = "smolry_yolo11s_640.onnx"
HELMET_URL = (
    f"https://raw.githubusercontent.com/{HELMET_REPO}/"
    f"{HELMET_COMMIT}/models/{HELMET_FILENAME}"
)
PLATE_REPO = "ml-debi/yolov8-license-plate-detection"
PLATE_REVISION = "8062e8e86734e272c60fa9a819433327ebdb66d7"
COCO_REPO = "kshitijjjjjjjjjjjjjjjj/yolov8n-coco-onnx"
COCO_REVISION = "7fdfbb82e54ed6b3c70722884f563450622b30fd"
COCO_CLASS_NAMES = {0: "person", 3: "motorcycle"}
HELMET_CLASS_NAMES = {0: "helmet", 1: "no-helmet"}
PLATE_CLASS_NAMES = {0: "license_plate"}
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _helmet_kind(class_name: str) -> str | None:
    normalized = _normalized_name(class_name)
    if normalized in {"nohelmet", "withouthelmet", "helmetless", "nohelmetrider"}:
        return "no_helmet"
    if normalized in {"helmet", "withhelmet", "helmetrider"}:
        return "helmet"
    return None


def _resolve_hub_model(
    supplied_path: Path | None,
    repo_id: str,
    filename: str,
    revision: str,
) -> str:
    if supplied_path is not None:
        resolved = supplied_path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Model weights do not exist: {resolved}")
        return str(resolved)

    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
    )


def _resolve_helmet_model(model_dir: Path) -> Path:
    model_path = model_dir / HELMET_FILENAME
    if model_path.is_file():
        return model_path

    model_dir.mkdir(parents=True, exist_ok=True)
    partial_path = model_path.with_suffix(model_path.suffix + ".part")
    try:
        with requests.get(HELMET_URL, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with partial_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
        partial_path.replace(model_path)
    finally:
        if partial_path.exists():
            partial_path.unlink()
    return model_path


def _read_plate(reader: Any, frame: Any, box: tuple[int, int, int, int]) -> tuple[str, float]:
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "", 0.0

    readings, _ = reader(crop)
    if not readings:
        return "", 0.0

    pieces: list[str] = []
    confidences: list[float] = []
    for reading in readings:
        text = re.sub(r"[^A-Z0-9]", "", str(reading[1]).upper())
        if text:
            pieces.append(text)
            confidences.append(float(reading[2]))
    return "".join(pieces), sum(confidences) / len(confidences) if confidences else 0.0


def _clamp_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x1 = max(0, min(width, int(box[0])))
    y1 = max(0, min(height, int(box[1])))
    x2 = max(0, min(width, int(box[2])))
    y2 = max(0, min(height, int(box[3])))
    return x1, y1, x2, y2


def _person_context_box(
    person: Detection, width: int, height: int
) -> tuple[int, int, int, int]:
    person_width = max(person.box[2] - person.box[0], 1.0)
    person_height = max(person.box[3] - person.box[1], 1.0)
    return (
        max(0, int(person.box[0] - person_width * 3)),
        max(0, int(person.box[1] - person_height * 0.25)),
        min(width, int(person.box[2] + person_width * 3)),
        min(height, int(person.box[3] + person_height * 1.25)),
    )


def _detect_person_heads(
    detector: OnnxDetector,
    frame: Any,
    people: list[Detection],
    confidence: float,
) -> list[Detection]:
    frame_height, frame_width = frame.shape[:2]
    detections: list[Detection] = []
    for person in people:
        x1, y1, x2, y2 = _person_context_box(
            person, frame_width, frame_height
        )
        if x1 >= x2 or y1 >= y2:
            continue
        for detection in detector.detect(frame[y1:y2, x1:x2], confidence):
            detections.append(
                Detection(
                    box=(
                        detection.box[0] + x1,
                        detection.box[1] + y1,
                        detection.box[2] + x1,
                        detection.box[3] + y1,
                    ),
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                )
            )
    return suppress_duplicates(detections)


def _motorcycle_rider_candidates(
    people: list[Detection],
    motorcycles: list[Detection],
) -> list[Detection]:
    return [
        person
        for person in people
        if match_person_to_motorcycle(person, motorcycles) is not None
    ]


def _associate_no_helmet_heads(
    heads: list[Detection],
    people: list[Detection],
    motorcycles: list[Detection],
) -> tuple[
    dict[tuple[str, int], Detection],
    dict[int, Detection],
    set[int],
    dict[int, int],
]:
    violation_subjects: dict[tuple[str, int], Detection] = {}
    matched_people: dict[int, Detection] = {}
    unmatched_people: set[int] = set()
    motorcycle_by_person: dict[int, int] = {}
    for head in heads:
        person = match_head_to_person(head, people)
        if person is None or person.track_id is None:
            continue
        matched_people[person.track_id] = person
        motorcycle = match_person_to_motorcycle(person, motorcycles)
        if motorcycle is None or motorcycle.track_id is None:
            unmatched_people.add(person.track_id)
            continue
        motorcycle_by_person[person.track_id] = motorcycle.track_id
        violation_subjects[("motorcycle", motorcycle.track_id)] = motorcycle
    return violation_subjects, matched_people, unmatched_people, motorcycle_by_person


def _draw_box(
    frame: Any,
    box: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = (int(value) for value in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def _save_reports(
    output_dir: Path,
    stem: str,
    source: Path,
    video_path: Path,
    ledger: EventLedger,
    summary: dict[str, int],
    frames_processed: int,
) -> tuple[Path, Path]:
    csv_path = output_dir / f"{stem}_violations.csv"
    json_path = output_dir / f"{stem}_violations.json"
    records = [event.as_dict() for event in ledger.confirmed_events]

    fieldnames = [
        "subject_track_id",
        "subject_type",
        "first_frame",
        "last_frame",
        "evidence_frames",
        "plate_text",
        "plate_confidence",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    report = {
        "source_video": str(source),
        "annotated_video": str(video_path),
        "event_count": len(records),
        "events": records,
        "frames_processed": frames_processed,
        "detection_summary": summary,
        "model_sources": {
            "helmet_no_helmet": f"https://github.com/{HELMET_REPO}/tree/{HELMET_COMMIT}/models",
            "license_plate": (
                f"https://huggingface.co/{PLATE_REPO}/tree/{PLATE_REVISION}"
            ),
            "person_motorcycle": (
                f"https://huggingface.co/{COCO_REPO}/tree/{COCO_REVISION}"
            ),
            "helmet_inference": "Per-person context crops",
        },
        "notice": (
            "ANPR reads are best-effort OCR results and require human verification. "
            "Detection performance depends on camera angle, image quality, and local plates."
        ),
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    return csv_path, json_path


def run_video(
    *,
    source: Path,
    output_dir: Path,
    helmet_model_path: Path | None,
    plate_model_path: Path | None,
    vehicle_model_path: Path | None,
    confidence: float,
    min_violation_frames: int,
    persistence_gap: int,
    ocr_interval: int,
    show: bool,
) -> None:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from rapidocr_onnxruntime import RapidOCR

    project_root = Path(__file__).resolve().parents[2]
    model_dir = project_root / "models"
    helmet_weights = (
        helmet_model_path.expanduser().resolve()
        if helmet_model_path is not None
        else _resolve_helmet_model(model_dir)
    )
    if not helmet_weights.is_file():
        raise FileNotFoundError(f"Helmet ONNX model does not exist: {helmet_weights}")

    plate_weights = _resolve_hub_model(
        plate_model_path,
        PLATE_REPO,
        "best.onnx",
        PLATE_REVISION,
    )
    vehicle_weights = _resolve_hub_model(
        vehicle_model_path,
        COCO_REPO,
        "yolov8n.onnx",
        COCO_REVISION,
    )

    helmet_detector = OnnxDetector(
        helmet_weights,
        HELMET_CLASS_NAMES,
        end_to_end=True,
        pad_value=0,
    )
    plate_detector = OnnxDetector(
        plate_weights,
        PLATE_CLASS_NAMES,
    )
    vehicle_detector = OnnxDetector(
        vehicle_weights,
        COCO_CLASS_NAMES,
    )
    ocr_reader = RapidOCR()

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Video has invalid frame dimensions: {width}x{height}")

    stem = source.stem
    video_path = output_dir / f"{stem}_annotated.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {video_path}")

    ledger = EventLedger(min_violation_frames, persistence_gap)
    person_tracker = ObjectTracker(max_gap=persistence_gap)
    motorcycle_tracker = ObjectTracker(max_gap=persistence_gap)
    frame_number = 0
    last_ocr_frame: dict[tuple[str, int], int] = {}
    summary = {
        "helmet_detections": 0,
        "no_helmet_detections": 0,
        "person_detections": 0,
        "motorcycle_detections": 0,
        "plate_detections": 0,
        "no_helmet_person_matches": 0,
        "unmatched_no_helmet_people": 0,
        "no_helmet_motorcycle_matches": 0,
        "frames_with_any_detection": 0,
    }
    try:
        while True:
            read_ok, frame = cap.read()
            if not read_ok:
                break
            frame_number += 1

            vehicle_detections = vehicle_detector.detect_quadrants(
                frame,
                confidence,
                class_ids=COCO_CLASS_NAMES,
            )
            people = person_tracker.update(
                [
                    detection
                    for detection in vehicle_detections
                    if detection.class_name == "person"
                ],
                frame_number,
            )
            motorcycles = motorcycle_tracker.update(
                [
                    detection
                    for detection in vehicle_detections
                    if detection.class_name == "motorcycle"
                ],
                frame_number,
            )
            rider_candidates = _motorcycle_rider_candidates(people, motorcycles)
            helmet_detections = _detect_person_heads(
                helmet_detector,
                frame,
                rider_candidates,
                confidence,
            )
            no_helmet_heads = [
                detection
                for detection in helmet_detections
                if _helmet_kind(detection.class_name) == "no_helmet"
            ]

            (
                violation_subjects,
                matched_people,
                unmatched_no_helmet_people,
                motorcycle_by_person,
            ) = _associate_no_helmet_heads(
                no_helmet_heads,
                rider_candidates,
                motorcycles,
            )
            plate_detections: list[Detection] = []
            plates_by_track: dict[int, Detection] = {}
            if violation_subjects:
                plate_detections = plate_detector.detect_tiled(
                    frame,
                    confidence,
                    class_ids=PLATE_CLASS_NAMES,
                )
                for plate in plate_detections:
                    motorcycle = match_plate_to_motorcycle(plate, motorcycles)
                    if motorcycle is not None and motorcycle.track_id is not None:
                        existing = plates_by_track.get(motorcycle.track_id)
                        if existing is None or plate.confidence > existing.confidence:
                            plates_by_track[motorcycle.track_id] = plate

            confirmed_for_frame: set[tuple[str, int]] = set()
            events_for_frame: dict[tuple[str, int], ViolationEvent] = {}
            for subject_key in violation_subjects:
                subject_type, track_id = subject_key
                plate = plates_by_track.get(track_id)
                plate_text = ""
                plate_score = 0.0
                if plate is not None and (
                    subject_key not in last_ocr_frame
                    or frame_number - last_ocr_frame[subject_key] >= ocr_interval
                ):
                    plate_box = _clamp_box(plate.box, width, height)
                    plate_text, plate_score = _read_plate(ocr_reader, frame, plate_box)
                    plate_score *= plate.confidence
                    last_ocr_frame[subject_key] = frame_number

                event = ledger.update(
                    track_id,
                    frame_number,
                    plate_text=plate_text,
                    plate_confidence=plate_score,
                    subject_type=subject_type,
                )
                events_for_frame[subject_key] = event
                if event.evidence_frames >= min_violation_frames:
                    confirmed_for_frame.add(subject_key)

            helmet_status_by_person: dict[int, str] = {}
            for head in helmet_detections:
                kind = _helmet_kind(head.class_name)
                if kind is not None:
                    person = match_head_to_person(head, rider_candidates)
                    if person is not None and person.track_id is not None:
                        current_kind = helmet_status_by_person.get(person.track_id)
                        if kind == "no_helmet" or current_kind is None:
                            helmet_status_by_person[person.track_id] = kind
                    motorcycle_id = (
                        motorcycle_by_person.get(person.track_id)
                        if person is not None and person.track_id is not None
                        else None
                    )
                    subject_key = (
                        ("motorcycle", motorcycle_id)
                        if motorcycle_id is not None
                        else None
                    )
                    if kind == "helmet":
                        color = (0, 200, 0)
                    elif subject_key in confirmed_for_frame:
                        color = (0, 0, 255)
                    else:
                        color = (0, 165, 255)
                    _draw_box(frame, head.box, "", color)

            for person in people:
                track_id = person.track_id
                subject_key = ("person", track_id) if track_id is not None else None
                motorcycle_id = (
                    motorcycle_by_person.get(track_id)
                    if track_id is not None
                    else None
                )
                violation_key = (
                    ("motorcycle", motorcycle_id)
                    if motorcycle_id is not None
                    else subject_key
                )
                if violation_key is not None and violation_key in events_for_frame:
                    event = events_for_frame[violation_key]
                    status = (
                        "NO HELMET"
                        if violation_key in confirmed_for_frame
                        else f"CHECK {event.evidence_frames}/{min_violation_frames}"
                    )
                    label = f"{status} | Person ID {track_id} | {event.plate_text or 'plate unread'}"
                    color = (
                        (0, 0, 255)
                        if violation_key in confirmed_for_frame
                        else (0, 165, 255)
                    )
                elif track_id in unmatched_no_helmet_people:
                    label = f"NO-HELMET CANDIDATE | NO BIKE MATCH | ID {track_id}"
                    color = (0, 165, 255)
                elif track_id in helmet_status_by_person and (
                    helmet_status_by_person[track_id] == "helmet"
                ):
                    label = f"HELMET | Person {person.confidence:.2f} | ID {track_id}"
                    color = (0, 200, 0)
                else:
                    label = f"Person {person.confidence:.2f} | ID {track_id}"
                    color = (0, 165, 255) if track_id in matched_people else (255, 200, 0)
                _draw_box(frame, person.box, label, color)

            for motorcycle in motorcycles:
                track_id = motorcycle.track_id
                subject_key = ("motorcycle", track_id) if track_id is not None else None
                if subject_key is not None and subject_key in events_for_frame:
                    event = events_for_frame[subject_key]
                    status = (
                        "NO HELMET"
                        if subject_key in confirmed_for_frame
                        else f"CHECK {event.evidence_frames}/{min_violation_frames}"
                    )
                    plate_label = event.plate_text or "plate unread"
                    color = (
                        (0, 0, 255)
                        if subject_key in confirmed_for_frame
                        else (0, 165, 255)
                    )
                    label = f"{status} | ID {track_id} | {plate_label}"
                else:
                    color = (255, 0, 0)
                    label = f"Motorcycle | ID {track_id} | {motorcycle.confidence:.2f}"
                _draw_box(
                    frame,
                    motorcycle.box,
                    label,
                    color,
                )

            for plate in plate_detections:
                motorcycle = match_plate_to_motorcycle(plate, motorcycles)
                if motorcycle is not None:
                    plate_label = f"Plate | Motorcycle ID {motorcycle.track_id}"
                else:
                    plate_label = "License plate"
                _draw_box(frame, plate.box, plate_label, (255, 180, 0))

            per_frame_counts = {
                "helmet_detections": sum(
                    _helmet_kind(detection.class_name) == "helmet"
                    for detection in helmet_detections
                ),
                "no_helmet_detections": len(no_helmet_heads),
                "person_detections": len(people),
                "motorcycle_detections": len(motorcycles),
                "plate_detections": len(plate_detections),
                "no_helmet_person_matches": len(matched_people),
                "unmatched_no_helmet_people": len(unmatched_no_helmet_people),
                "no_helmet_motorcycle_matches": sum(
                    subject_type == "motorcycle"
                    for subject_type, _ in violation_subjects
                ),
            }
            detected_this_frame = any(per_frame_counts.values())
            if detected_this_frame:
                summary["frames_with_any_detection"] += 1
            for name, count in per_frame_counts.items():
                summary[name] += count
            if not detected_this_frame:
                frame_status = "No objects detected"
                status_color = (0, 0, 255)
            else:
                status_color = (0, 255, 255)
                frame_status = (
                    f"H:{per_frame_counts['helmet_detections']} "
                    f"NH:{per_frame_counts['no_helmet_detections']} "
                    f"P:{per_frame_counts['person_detections']} "
                    f"M:{per_frame_counts['motorcycle_detections']} "
                    f"PL:{per_frame_counts['plate_detections']}"
                )
            cv2.putText(
                frame,
                frame_status,
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                status_color,
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            if show:
                cv2.imshow("No-helmet detection + ANPR", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    csv_path, json_path = _save_reports(
        output_dir,
        stem,
        source,
        video_path,
        ledger,
        summary,
        frame_number,
    )
    print(f"Annotated video: {video_path}")
    print(f"Violation CSV:   {csv_path}")
    print(f"Violation JSON:  {json_path}")
    print(f"Confirmed events: {len(ledger.confirmed_events)}")
    print(f"Detection summary: {summary}")
