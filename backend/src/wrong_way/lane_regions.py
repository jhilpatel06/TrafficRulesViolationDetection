# backend/src/wrong_way/lane_regions.py
#--------------------------------------------------------------------------------
# Loading CVAT polygon annotations.
# Identifying the road region containing a vehicle.
# Determining the expected direction of movement.
# Supporting wrong-way violation detection.


import cv2
import numpy as np


class LaneRegions:
    def __init__(self, lanes):
        self.lanes = lanes

    def get_lane(self, bbox):
        x1, y1, x2, y2 = bbox

        # Bottom-centre point of the vehicle
        point = (
            int((x1 + x2) / 2),
            int(y2)
        )

        for lane in self.lanes:
            polygon = np.array(
                lane["polygon"],
                dtype=np.int32
            )

            result = cv2.pointPolygonTest(
                polygon,
                point,
                False
            )

            if result >= 0:
                return lane

        return None
