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
from .onnx_models import OnnxDetector
from .tracking import MotorcycleTracker


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


def _draw_box(
    frame: Any,
    box: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = (int(value) for value in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
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
) -> tuple[Path, Path]:
    csv_path = output_dir / f"{stem}_violations.csv"
    json_path = output_dir / f"{stem}_violations.json"
    records = [event.as_dict() for event in ledger.confirmed_events]

    fieldnames = [
        "motorcycle_track_id",
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
        "model_sources": {
            "helmet_no_helmet": f"https://github.com/{HELMET_REPO}/tree/{HELMET_COMMIT}/models",
            "license_plate": (
                f"https://huggingface.co/{PLATE_REPO}/tree/{PLATE_REVISION}"
            ),
            "person_motorcycle": (
                f"https://huggingface.co/{COCO_REPO}/tree/{COCO_REVISION}"
            ),
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
    tracker = MotorcycleTracker(max_gap=persistence_gap)
    frame_number = 0
    last_ocr_frame: dict[int, int] = {}
    try:
        while True:
            read_ok, frame = cap.read()
            if not read_ok:
                break
            frame_number += 1

            helmet_detections = helmet_detector.detect(frame, confidence)
            no_helmet_heads = [
                detection
                for detection in helmet_detections
                if _helmet_kind(detection.class_name) == "no_helmet"
            ]

            vehicle_detections = vehicle_detector.detect(
                frame,
                confidence,
                class_ids=COCO_CLASS_NAMES,
            )
            people = [
                detection
                for detection in vehicle_detections
                if detection.class_name == "person"
            ]
            motorcycles = tracker.update(
                [
                    detection
                    for detection in vehicle_detections
                    if detection.class_name == "motorcycle"
                ],
                frame_number,
            )

            plate_detections = plate_detector.detect(
                frame,
                confidence,
                class_ids=PLATE_CLASS_NAMES,
            )
            plates_by_track: dict[int, Detection] = {}
            for plate in plate_detections:
                motorcycle = match_plate_to_motorcycle(plate, motorcycles)
                if motorcycle is None or motorcycle.track_id is None:
                    continue
                existing = plates_by_track.get(motorcycle.track_id)
                if existing is None or plate.confidence > existing.confidence:
                    plates_by_track[motorcycle.track_id] = plate

            violation_motorcycles: dict[int, Detection] = {}
            for head in no_helmet_heads:
                person = match_head_to_person(head, people)
                if person is None:
                    continue
                motorcycle = match_person_to_motorcycle(person, motorcycles)
                if motorcycle is not None and motorcycle.track_id is not None:
                    violation_motorcycles[motorcycle.track_id] = motorcycle

            confirmed_for_frame: dict[int, ViolationEvent] = {}
            events_for_frame: dict[int, ViolationEvent] = {}
            for track_id in violation_motorcycles:
                plate = plates_by_track.get(track_id)
                plate_text = ""
                plate_score = 0.0
                if plate is not None and (
                    track_id not in last_ocr_frame
                    or frame_number - last_ocr_frame[track_id] >= ocr_interval
                ):
                    plate_box = _clamp_box(plate.box, width, height)
                    plate_text, plate_score = _read_plate(ocr_reader, frame, plate_box)
                    plate_score *= plate.confidence
                    last_ocr_frame[track_id] = frame_number

                event = ledger.update(
                    track_id,
                    frame_number,
                    plate_text=plate_text,
                    plate_confidence=plate_score,
                )
                events_for_frame[track_id] = event
                if event.evidence_frames >= min_violation_frames:
                    confirmed_for_frame[track_id] = event

            for motorcycle in motorcycles:
                track_id = motorcycle.track_id
                if track_id not in violation_motorcycles:
                    continue
                event = events_for_frame[track_id]
                status = (
                    "NO HELMET"
                    if track_id in confirmed_for_frame
                    else f"CHECK {event.evidence_frames}/{min_violation_frames}"
                )
                plate_label = event.plate_text or "plate unread"
                _draw_box(
                    frame,
                    motorcycle.box,
                    f"{status} | ID {track_id} | {plate_label}",
                    (0, 0, 255) if track_id in confirmed_for_frame else (0, 165, 255),
                )

            for head in no_helmet_heads:
                _draw_box(frame, head.box, "No helmet", (0, 165, 255))
            for track_id, plate in plates_by_track.items():
                _draw_box(frame, plate.box, f"Plate | ID {track_id}", (255, 180, 0))

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
        output_dir, stem, source, video_path, ledger
    )
    print(f"Annotated video: {video_path}")
    print(f"Violation CSV:   {csv_path}")
    print(f"Violation JSON:  {json_path}")
    print(f"Confirmed events: {len(ledger.confirmed_events)}")
