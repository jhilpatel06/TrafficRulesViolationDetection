# backend/src/wrong_way/visualize_debug.py
"""
Debug overlay: lane polygons, direction arrows, vehicle centroids,
bottom-center anchors, and per-vehicle lane-membership status.
"""

import csv
import cv2
import numpy as np

from wrong_way_detection.backend.src.wrong_way.lane_regions import LaneRegions
from wrong_way_detection.backend.src.helper import load_config

# ------------------- Helpers ---------------------------------
def _extract_polygon(lane):
    """Return the polygon of a lane dict as an (N, 2) int32 array.

    Tries the common key names used by typical config files.
    """
    for key in ("polygon", "polygons", "points", "vertices",
                "contour", "region", "coords"):
        if key in lane:
            return np.array(lane[key], dtype=np.int32)
    raise KeyError(
        f"Lane dict has no polygon key. Available keys: {list(lane.keys())}"
    )


def _lane_label(lane, index):
    """Human readable label for a lane dict."""
    for key in ("name", "id", "label", "lane_id"):
        if key in lane:
            return str(lane[key])
    return f"L{index}"


def _lane_centroid(poly):
    return tuple(np.mean(poly, axis=0).astype(int))


def _draw_lanes(canvas, lanes_cfg):
    for i, lane in enumerate(lanes_cfg):
        poly = _extract_polygon(lane)
        label = _lane_label(lane, i)

        # outline only — no fill
        cv2.polylines(canvas, [poly], True, (255, 0, 255), 2)

        cx, cy = _lane_centroid(poly)
        cv2.putText(canvas, label, (cx - 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # permitted direction arrow
        if "direction" in lane:
            d = np.array(lane["direction"], dtype=float)
            n = np.linalg.norm(d)
            if n > 1e-6:
                d = d / n * 50.0
                cv2.arrowedLine(
                    canvas, (cx, cy),
                    (int(cx + d[0]), int(cy + d[1])),
                    (255, 0, 255), 2, tipLength=0.35,
                )

def _draw_track(canvas, track, lane_regions, csv_rows, frame_idx):
    x1, y1, x2, y2 = map(int, track.bbox)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    anchor = (cx, y2)

    # NOTE: in this codebase, get_lane(bbox) and get_lane_at_point(anchor)
    # are equivalent — get_lane() already uses the bottom-centre point.
    lane_from_box = lane_regions.get_lane(track.bbox)
    lane_from_anchor = lane_regions.get_lane_at_point(anchor)

    color = (0, 255, 0) if lane_from_box is not None else (0, 0, 255)

    # bbox
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    # centroid (yellow) and bottom-center anchor (cyan)
    cv2.circle(canvas, (cx, cy), 5, (0, 255, 255), -1)
    cv2.circle(canvas, anchor, 5, (255, 255, 0), -1)
    cv2.line(canvas, (cx, cy), anchor, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"ID {track.track_id}", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(canvas,
                f"box:{'IN' if lane_from_box else 'OUT'}",
                (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 0) if lane_from_box else (0, 0, 255), 1)
    cv2.putText(canvas,
                f"anc:{'IN' if lane_from_anchor else 'OUT'}",
                (x1, y2 + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 0) if lane_from_anchor else (0, 0, 255), 1)

    csv_rows.append({
        "frame": frame_idx,
        "track_id": track.track_id,
        "class": getattr(track, "class_name", ""),
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "cx": cx, "cy": cy,
        "ax": anchor[0], "ay": anchor[1],
        "lane_from_box": lane_from_box is not None,
        "lane_from_anchor": lane_from_anchor is not None,
    })

# ---------- public API ----------

def annotate_frame(frame, tracks, lane_regions, lanes_cfg, frame_idx, csv_rows):
    """Return a debug-annotated copy of *frame* for one video frame."""
    canvas = frame.copy()
    _draw_lanes(canvas, lanes_cfg)
    for track in tracks:
        _draw_track(canvas, track, lane_regions, csv_rows, frame_idx)
    return canvas


def run_debug_video(video_path, output_path, tracker, csv_path=None, show=False):
    """Run the tracker across the whole video and write an annotated copy.

    Parameters
    ----------
    video_path : str
        Input video file.
    output_path : str
        Annotated output video (mp4).
    tracker : object
        Your tracker instance; must expose ``update(frame) -> list[Track]``
        where each Track has ``track_id``, ``bbox`` and optionally
        ``class_name``.
    csv_path : str | None
        Optional path to dump per-frame centroid/anchor membership CSV.
    show : bool
        If True, display frames with cv2.imshow (press q to quit).
    """
    config = load_config()
    lanes_cfg = config["lane_directions"]["lanes"]
    lane_regions = LaneRegions(lanes_cfg)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    csv_rows = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        tracks = tracker.update(frame)
        canvas = annotate_frame(
            frame, tracks, lane_regions, lanes_cfg, frame_idx, csv_rows
        )

        if writer is None:
            h, w = canvas.shape[:2]
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (w, h),
            )
        writer.write(canvas)

        if show:
            cv2.imshow("lane-debug", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(csv_rows)

    print(f"Processed {frame_idx} frames → {output_path}")