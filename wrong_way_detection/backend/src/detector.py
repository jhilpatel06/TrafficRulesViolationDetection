# backend/src/detection/vehicle_detector.py

from ultralytics import YOLO


class VehicleDetector:
    def __init__(
        self,
        confidence: float,
        iou: float,
        vehicle_class_ids: list[int] | None = None,
        device: str = "cpu",
        imgsz: int = 640,
    ):

        # Load model once during initialization
        self.model = YOLO("yolo26s.pt")
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.vehicle_class_ids = vehicle_class_ids
        self.imgsz = imgsz

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
            imgsz=self.imgsz,
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

    @property
    def allowed_class_names(self) -> set[str]:
        """Class-name strings corresponding to `vehicle_class_ids`.

        Returns every YOLO class name if `vehicle_class_ids` is None,
        otherwise the subset named by the config.
        """
        if self.vehicle_class_ids is None:
            return set(self.model.names.values())
        return {self.model.names[i] for i in self.vehicle_class_ids}
