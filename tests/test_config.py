from pathlib import Path

import pytest

from alpr.config import PipelineConfig, load_config, parse_overrides

REPO = Path(__file__).parents[1]


def test_defaults_are_valid():
    PipelineConfig().validate()


def test_documented_yaml_matches_the_built_in_defaults():
    assert load_config(REPO / "configs" / "pipeline.yaml") == PipelineConfig()


def test_overrides_keep_their_types():
    config = load_config(
        overrides=[
            "tracker.max_age=7",
            "plate.conf=0.4",
            "ocr.backend=fast-plate-ocr",
            "plate_search=vehicle",
            "min_plate_overlap=1",  # an int is fine where a float is expected
        ]
    )
    assert config.tracker.max_age == 7
    assert config.plate.conf == 0.4
    assert config.ocr.backend == "fast-plate-ocr"
    assert config.plate_search == "vehicle"
    assert config.min_plate_overlap == 1.0 and isinstance(config.min_plate_overlap, float)


def test_partial_override_keeps_the_other_defaults():
    config = load_config(overrides=["vehicle.conf=0.5"])
    assert config.vehicle.weights == PipelineConfig().vehicle.weights
    assert config.vehicle.imgsz == 640
    assert config.plate == PipelineConfig().plate


def test_file_then_overrides(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("tracker:\n  max_age: 12\n  min_hits: 2\n")
    config = load_config(path, ["tracker.max_age=3"])
    assert (config.tracker.max_age, config.tracker.min_hits) == (3, 2)
    assert load_config(tmp_path / "custom.yaml").tracker.iou_threshold == 0.3


def test_empty_file_gives_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("# nothing here\n")
    assert load_config(path) == PipelineConfig()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("tracker.max_agee=3", r"unknown config key\(s\): tracker.max_agee"),
        ("trakcer.max_age=3", r"unknown config key\(s\): trakcer"),
        ("tracker.max_age=soon", "tracker.max_age must be int, got str"),
        ("tracker.max_age=2.5", "tracker.max_age must be int, got float"),
        ("tracker.max_age=true", "tracker.max_age must be int, got bool"),
        ("plate.conf=high", "plate.conf must be float"),
        ("tracker=5", "tracker must be a mapping"),
        ("plate_search=satellite", "plate_search must be one of"),
        ("ocr.backend=tesseract", "ocr.backend must be one of"),
        ("ocr.preprocess=gray+sharpen", "unknown preprocessing"),
        ("ocr.every=0", "ocr.every must be at least 1"),
        ("plate_format.name=narnia", "unknown plate format"),
        ("aggregation.strategy=coin_flip", "aggregation.strategy must be one of"),
    ],
)
def test_mistakes_are_reported_by_name(override, message):
    with pytest.raises(ValueError, match=message):
        load_config(overrides=[override])


def test_parse_overrides():
    assert parse_overrides(["a.b.c=1", "a.b.d=x", "e=0.5", "f=true"]) == {
        "a": {"b": {"c": 1, "d": "x"}},
        "e": 0.5,
        "f": True,
    }
    for bad in ["novalue", "=3", ""]:
        with pytest.raises(ValueError, match="key=value"):
            parse_overrides([bad])


def test_to_dict_is_plain_and_nested():
    data = PipelineConfig().to_dict()
    assert data["tracker"] == {"max_age": 30, "min_hits": 3, "iou_threshold": 0.3}
    assert data["vehicle"]["weights"] == "weights/yolo11n.pt"
