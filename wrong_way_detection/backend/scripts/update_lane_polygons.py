# backend/scripts/update_lane_polygons.py

import xml.etree.ElementTree as ET
from pathlib import Path
import sys

try:
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedSeq
except ImportError:
    print("Error: ruamel.yaml is not installed.")
    print("Please run: pip install ruamel.yaml")
    sys.exit(1)

# --- Configuration Paths ---
# Adjust these paths if your folder structure is different
ROOT_DIR = Path(__file__).resolve().parents[1]
XML_PATH = ROOT_DIR / "data" / "annotations.xml"
YAML_PATH = ROOT_DIR / "configs" / "configs.yaml"

# Mapping CVAT labels to your YAML lane IDs
LABEL_TO_LANE_ID = {
    "up": "right",
    "Down": "left"
}

def parse_cvat_points(points_str):
    """Converts CVAT point string 'x1,y1;x2,y2' to nested list [[x1, y1], [x2, y2]]"""
    polygon = []
    for pair in points_str.split(';'):
        x, y = pair.split(',')
        point = CommentedSeq([int(float(x)), int(float(y))])
        point.fa.set_flow_style()
        polygon.append(point)
    return polygon

def main():

    # 1. Parse XML and extract coordinates
    tree = ET.parse(XML_PATH)
    root = tree.getroot()
    
    new_polygons = {}
    for track in root.findall("track"):
        label = track.get("label")
        if label in LABEL_TO_LANE_ID:
            # Grab the very first frame's polygon (keyframe)
            first_polygon = track.find("polygon")
            if first_polygon is not None:
                points_str = first_polygon.get("points")
                lane_id = LABEL_TO_LANE_ID[label]
                new_polygons[lane_id] = parse_cvat_points(points_str)
                print(f"Extracted {len(new_polygons[lane_id])} points for '{lane_id}' lane.")

    # 2. Load YAML file (preserving comments)
    yaml = YAML()
    yaml.preserve_quotes = True
    
    with YAML_PATH.open("r", encoding="utf-8") as file:
        config = yaml.load(file)

    # 3. Inject new coordinates
    updated_count = 0
    for lane in config.get("lane_directions", {}).get("lanes", []):
        lane_id = lane.get("id")
        if lane_id in new_polygons:
            lane["polygon"] = new_polygons[lane_id]
            updated_count += 1

    # 4. Save updated YAML back to disk
    if updated_count > 0:
        with YAML_PATH.open("w", encoding="utf-8") as file:
            yaml.dump(config, file)
        print(f"\nSuccess! Updated {updated_count} lane(s) in {YAML_PATH.name}")
    else:
        print("\nNo matching lanes were found to update in the YAML file.")

if __name__ == "__main__":
    main()