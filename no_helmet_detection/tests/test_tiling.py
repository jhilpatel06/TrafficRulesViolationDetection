from src.no_helmet_detection.onnx_models import _quadrant_regions, _tile_regions


def test_high_resolution_tiles_overlap_and_cover_all_image_corners() -> None:
    regions = _tile_regions(1908, 1070)

    assert len(regions) == 2
    assert regions[0][0:2] == (0, 0)
    assert regions[0][3] == 1070
    assert regions[1][2:] == (1908, 1070)
    assert regions[0][2] > regions[1][0]


def test_tall_frame_is_split_into_overlapping_horizontal_tiles() -> None:
    regions = _tile_regions(1080, 1920)

    assert len(regions) == 2
    assert regions[0][:3:2] == (0, 1080)
    assert regions[1][0] == 0
    assert regions[1][2] == 1080
    assert regions[0][3] > regions[1][1]


def test_vehicle_quadrants_overlap_and_cover_all_image_corners() -> None:
    regions = _quadrant_regions(1908, 1070)

    assert len(regions) == 4
    assert regions[0][:2] == (0, 0)
    assert regions[1][2] == 1908
    assert regions[2][3] == 1070
    assert regions[3][2:] == (1908, 1070)
    assert regions[0][2] > regions[1][0]
    assert regions[0][3] > regions[2][1]
