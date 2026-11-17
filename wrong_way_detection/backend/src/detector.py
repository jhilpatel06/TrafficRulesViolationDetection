# backend/src/vehicle_detector.py
# --------------------------------------------------------------------------------
# loads YOLOv5 weights and returns vehicle detections in a predictable format.

from pathlib import Path
import torch


class VehicleDetector:
    def __init__(
        self,
        weights: str,
        yolov5_repo: str,
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        vehicle_class_ids: list[int] | None = None,
        device: str = "cpu",
    ):
        """
        weights: path to your trained .pt weights
        yolov5_repo: local path to the cloned YOLOv5 repository
        vehicle_class_ids:
            None -> keep all classes from your model
            COCO vehicle IDs -> [2, 3, 5, 7]
        """
        repo_path = Path(yolov5_repo)

        if not (repo_path / "hubconf.py").exists():
            raise FileNotFoundError(
                f"YOLOv5 repository not found at: {repo_path}"
            )

        if not Path(weights).exists():
            raise FileNotFoundError(f"Weights not found: {weights}")

        self.model = torch.hub.load(
            yolo26n.pt,
            "custom",
            path=weights,
            source="local",
        )

        self.model.conf = confidence
        self.model.iou = iou
        self.model.to(device)

        self.image_size = image_size
        self.vehicle_class_ids = (
            set(vehicle_class_ids)
            if vehicle_class_ids is not None
            else None
        )

    def detect(self, frame):
        """
        Returns:
        [
            {
                "bbox": (x1, y1, x2, y2),
                "confidence": 0.91,
                "class_id": 2,
                "class_name": "car"
            },
            ...
        ]
        """
        results = self.model(frame, size=self.image_size)
        predictions = results.xyxy[0].detach().cpu().numpy()

        detections = []

        for x1, y1, x2, y2, confidence, class_id in predictions:
            class_id = int(class_id)

            if (
                self.vehicle_class_ids is not None
                and class_id not in self.vehicle_class_ids
            ):
                continue

            detections.append({
                "bbox": (
                    int(x1), int(y1), int(x2), int(y2)
                ),
                "confidence": float(confidence),
                "class_id": class_id,
                "class_name": self.model.names[class_id],
            })

        return detections