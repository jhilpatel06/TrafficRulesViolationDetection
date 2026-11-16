from src.no_helmet_detection.pipeline import _helmet_kind


def test_helmet_model_labels_are_normalized() -> None:
    assert _helmet_kind("No-Helmet") == "no_helmet"
    assert _helmet_kind("with_helmet") == "helmet"
    assert _helmet_kind("motorcycle") is None
