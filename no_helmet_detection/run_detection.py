from __future__ import annotations

import argparse
from pathlib import Path

from src.no_helmet_detection.pipeline import run_video


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect motorcycle riders without helmets and associate ANPR reads."
    )
    parser.add_argument("source", type=Path, help="Input CCTV video file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Where annotated video and CSV/JSON reports are written.",
    )
    parser.add_argument(
        "--helmet-model",
        type=Path,
        help="Optional local YOLO weights; otherwise the pinned Hugging Face model is downloaded.",
    )
    parser.add_argument(
        "--plate-model",
        type=Path,
        help="Optional local YOLO/ONNX weights; otherwise the pinned Hugging Face model is downloaded.",
    )
    parser.add_argument(
        "--vehicle-model",
        type=Path,
        help="Optional local COCO-compatible ONNX weights for person/motorcycle detection.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum confidence for helmet and plate detections.",
    )
    parser.add_argument(
        "--min-violation-frames",
        type=int,
        default=3,
        help="Evidence frames required before a violation is reported.",
    )
    parser.add_argument(
        "--persistence-gap",
        type=int,
        default=2,
        help="Maximum missing frames between evidence observations for one event.",
    )
    parser.add_argument(
        "--ocr-interval",
        type=int,
        default=5,
        help="Run OCR at most once per this many frames for each current violation.",
    )
    parser.add_argument("--show", action="store_true", help="Display the annotated video.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 < args.confidence <= 1:
        raise SystemExit("--confidence must be greater than 0 and at most 1.")
    if args.min_violation_frames < 1:
        raise SystemExit("--min-violation-frames must be at least 1.")
    if args.persistence_gap < 0:
        raise SystemExit("--persistence-gap cannot be negative.")
    if args.ocr_interval < 1:
        raise SystemExit("--ocr-interval must be at least 1.")

    run_video(
        source=args.source,
        output_dir=args.output_dir,
        helmet_model_path=args.helmet_model,
        plate_model_path=args.plate_model,
        vehicle_model_path=args.vehicle_model,
        confidence=args.confidence,
        min_violation_frames=args.min_violation_frames,
        persistence_gap=args.persistence_gap,
        ocr_interval=args.ocr_interval,
        show=args.show,
    )


if __name__ == "__main__":
    main()
