from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort

from .geometry import Box, Detection


@dataclass(frozen=True)
class LetterboxTransform:
    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int
    original_width: int
    original_height: int


def _letterbox(
    frame: np.ndarray,
    width: int,
    height: int,
    pad_value: int,
) -> tuple[np.ndarray, LetterboxTransform]:
    original_height, original_width = frame.shape[:2]
    scale = min(width / original_width, height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    pad_left = (width - resized_width) // 2
    pad_top = (height - resized_height) // 2

    canvas = np.full((height, width, 3), pad_value, dtype=np.uint8)
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    canvas[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = resized

    tensor = canvas.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    transform = LetterboxTransform(
        scale_x=resized_width / original_width,
        scale_y=resized_height / original_height,
        pad_left=pad_left,
        pad_top=pad_top,
        original_width=original_width,
        original_height=original_height,
    )
    return tensor, transform


def _restore_box(
    box: tuple[float, float, float, float], transform: LetterboxTransform
) -> Box:
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(transform.original_width, (x1 - transform.pad_left) / transform.scale_x)),
        max(0.0, min(transform.original_height, (y1 - transform.pad_top) / transform.scale_y)),
        max(0.0, min(transform.original_width, (x2 - transform.pad_left) / transform.scale_x)),
        max(0.0, min(transform.original_height, (y2 - transform.pad_top) / transform.scale_y)),
    )


def _iou(first: Box, second: Box) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _nms(
    detections: list[Detection], iou_threshold: float
) -> list[Detection]:
    kept: list[Detection] = []
    for class_name in {detection.class_name for detection in detections}:
        remaining = sorted(
            (detection for detection in detections if detection.class_name == class_name),
            key=lambda detection: detection.confidence,
            reverse=True,
        )
        while remaining:
            best = remaining.pop(0)
            kept.append(best)
            remaining = [
                candidate
                for candidate in remaining
                if _iou(best.box, candidate.box) <= iou_threshold
            ]
    return kept


def suppress_duplicates(
    detections: list[Detection], iou_threshold: float = 0.45
) -> list[Detection]:
    return _nms(detections, iou_threshold)


def _tile_regions(width: int, height: int) -> list[tuple[int, int, int, int]]:
    if width >= height:
        tile_width = min(width, max(1, int(round(width * 0.55))))
        x_starts = sorted({0, max(0, width - tile_width)})
        return [(x, 0, min(width, x + tile_width), height) for x in x_starts]

    tile_height = min(height, max(1, int(round(height * 0.55))))
    y_starts = sorted({0, max(0, height - tile_height)})
    return [(0, y, width, min(height, y + tile_height)) for y in y_starts]


def _quadrant_regions(width: int, height: int) -> list[tuple[int, int, int, int]]:
    tile_width = min(width, max(1, int(round(width * 0.52))))
    tile_height = min(height, max(1, int(round(height * 0.52))))
    x_starts = sorted({0, max(0, width - tile_width)})
    y_starts = sorted({0, max(0, height - tile_height)})
    return [
        (x, y, min(width, x + tile_width), min(height, y + tile_height))
        for y in y_starts
        for x in x_starts
    ]


class OnnxDetector:
    def __init__(
        self,
        model_path: str | Path,
        class_names: dict[int, str],
        *,
        end_to_end: bool = False,
        pad_value: int = 114,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model does not exist: {model_path}")

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        model_input = self.session.get_inputs()[0]
        shape = model_input.shape
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError(f"Expected an ONNX NCHW image input, got shape {shape}")
        if not isinstance(shape[2], int) or not isinstance(shape[3], int):
            raise ValueError(f"Expected fixed ONNX image dimensions, got shape {shape}")

        self.input_name = model_input.name
        self.input_height = shape[2]
        self.input_width = shape[3]
        self.output_name = self.session.get_outputs()[0].name
        self.class_names = class_names
        self.end_to_end = end_to_end
        self.pad_value = pad_value

    def detect(
        self,
        frame: np.ndarray,
        confidence: float,
        class_ids: Iterable[int] | None = None,
    ) -> list[Detection]:
        tensor, transform = _letterbox(
            frame,
            self.input_width,
            self.input_height,
            self.pad_value,
        )
        output = self.session.run(
            [self.output_name],
            {self.input_name: tensor},
        )[0]
        selected_ids = set(class_ids) if class_ids is not None else set(self.class_names)

        if self.end_to_end:
            detections = self._decode_end_to_end(output, transform, confidence, selected_ids)
        else:
            detections = self._decode_yolo(output, transform, confidence, selected_ids)
        return _nms(detections, iou_threshold=0.45)

    def detect_tiled(
        self,
        frame: np.ndarray,
        confidence: float,
        class_ids: Iterable[int] | None = None,
    ) -> list[Detection]:
        height, width = frame.shape[:2]
        if width <= self.input_width and height <= self.input_height:
            return self.detect(frame, confidence, class_ids)
        return self._detect_regions(
            frame,
            _tile_regions(width, height),
            confidence,
            class_ids,
        )

    def detect_quadrants(
        self,
        frame: np.ndarray,
        confidence: float,
        class_ids: Iterable[int] | None = None,
    ) -> list[Detection]:
        height, width = frame.shape[:2]
        if width <= self.input_width and height <= self.input_height:
            return self.detect(frame, confidence, class_ids)
        return self._detect_regions(
            frame,
            _quadrant_regions(width, height),
            confidence,
            class_ids,
        )

    def _detect_regions(
        self,
        frame: np.ndarray,
        regions: list[tuple[int, int, int, int]],
        confidence: float,
        class_ids: Iterable[int] | None,
    ) -> list[Detection]:
        if len(regions) <= 1:
            return self.detect(frame, confidence, class_ids)

        height, width = frame.shape[:2]
        detections: list[Detection] = []
        for x1, y1, x2, y2 in regions:
            for detection in self.detect(
                frame[y1:y2, x1:x2],
                confidence,
                class_ids,
            ):
                box = (
                    max(0.0, min(float(width), detection.box[0] + x1)),
                    max(0.0, min(float(height), detection.box[1] + y1)),
                    max(0.0, min(float(width), detection.box[2] + x1)),
                    max(0.0, min(float(height), detection.box[3] + y1)),
                )
                detections.append(
                    Detection(
                        box=box,
                        class_name=detection.class_name,
                        confidence=detection.confidence,
                    )
                )
        return _nms(detections, iou_threshold=0.45)

    def _decode_end_to_end(
        self,
        output: np.ndarray,
        transform: LetterboxTransform,
        confidence: float,
        class_ids: set[int],
    ) -> list[Detection]:
        rows = np.asarray(output).reshape(-1, 6)
        detections = []
        for x1, y1, x2, y2, score, class_id in rows:
            class_index = int(class_id)
            if score < confidence or class_index not in class_ids:
                continue
            detections.append(
                Detection(
                    box=_restore_box((float(x1), float(y1), float(x2), float(y2)), transform),
                    class_name=self.class_names[class_index],
                    confidence=float(score),
                )
            )
        return detections

    def _decode_yolo(
        self,
        output: np.ndarray,
        transform: LetterboxTransform,
        confidence: float,
        class_ids: set[int],
    ) -> list[Detection]:
        predictions = np.asarray(output).squeeze(axis=0)
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        if predictions.shape[1] < 5:
            raise ValueError(f"Unexpected YOLO ONNX output shape: {output.shape}")

        valid_class_ids = sorted(
            class_id for class_id in class_ids if class_id < predictions.shape[1] - 4
        )
        if not valid_class_ids:
            raise ValueError(
                "Requested classes are not present in the YOLO output: "
                f"output shape={output.shape}, requested={sorted(class_ids)}"
            )

        class_scores = predictions[:, 4 + np.asarray(valid_class_ids)]
        best_offsets = np.argmax(class_scores, axis=1)
        best_scores = class_scores[np.arange(class_scores.shape[0]), best_offsets]
        selected = np.flatnonzero(best_scores >= confidence)
        detections = []
        for row_index in selected:
            class_id = valid_class_ids[int(best_offsets[row_index])]
            center_x, center_y, box_width, box_height = predictions[row_index, :4]
            box = _restore_box(
                (
                    float(center_x - box_width / 2),
                    float(center_y - box_height / 2),
                    float(center_x + box_width / 2),
                    float(center_y + box_height / 2),
                ),
                transform,
            )
            detections.append(
                Detection(
                    box=box,
                    class_name=self.class_names[class_id],
                    confidence=float(best_scores[row_index]),
                )
            )
        return detections
