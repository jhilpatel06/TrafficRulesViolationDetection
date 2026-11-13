# backend/src/wrong_way/lane_regions.py

import cv2
import numpy as np


class LaneRegions:
    def __init__(self, lanes):
        self.lanes = lanes

    def get_lane_at_point(self, point):
        """Return the lane dict containing `point` (x, y), else None."""
        px, py = int(point[0]), int(point[1])

        for lane in self.lanes:
            polygon = np.array(lane["polygon"], dtype=np.int32)

            result = cv2.pointPolygonTest(
                polygon,
                (px, py),
                False,
            )

            if result >= 0:
                return lane

        return None
    
    def get_lane(self, bbox):
        x1, y1, x2, y2 = bbox

        # Bottom-centre point of the vehicle
        point = (
            int((x1 + x2) / 2),
            int(y2)
        )

        return self.get_lane_at_point(point)