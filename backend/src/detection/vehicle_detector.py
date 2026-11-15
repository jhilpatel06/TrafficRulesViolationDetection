# backend/src/detection/vehicle_detector.py

from ultralytics import YOLO


class VehicleDetector:
    def __init__(
        self,
        weights: str,
        confidence: float,
        iou: float,
        vehicle_class_ids: list[int] | None = None,
        device: str = "cpu",
    ):

        # Load model once during initialization
        self.model = YOLO(str(weights))

        self.confidence = confidence
        self.iou = iou
        self.device = device

        self.vehicle_class_ids = vehicle_class_ids

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

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            classes=self.vehicle_class_ids,
            device=self.device,
            verbose=False
        )

        detections = []

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                detections.append({
                    "bbox": (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ),
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": self.model.names[class_id],
                })

        return detections
