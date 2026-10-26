import pytest
import yaml

from alpr.training import (
    config_differences,
    load_train_args,
    read_results_csv,
    summarize_training_run,
)

COLUMNS = [
    "epoch",
    "time",
    "train/box_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
]
EPOCHS = [
    [1, 70.5, 1.5, 0.80, 0.70, 0.75, 0.40, 1.6, 0.9, 1.4],
    [2, 141.0, 1.3, 0.95, 0.93, 0.97, 0.70, 1.1, 0.4, 1.1],
    [3, 212.4, 1.2, 0.97, 0.96, 0.98, 0.69, 1.0, 0.3, 1.1],
]
# Shaped like an Ultralytics results.csv, padded the way older releases pad it.
RESULTS_CSV = "".join(
    ",".join(f"{cell:>20}" for cell in line) + "\n" for line in [COLUMNS, *EPOCHS]
)

RUN_ARGS = {
    "task": "detect",
    "model": "yolo11n.pt",
    "data": "C:/Users/bart/data.yaml",
    "epochs": 3,
    "batch": 16,
    "imgsz": 640,
    "lr0": 0.01,
    "momentum": 0.937,
    "seed": 0,
    "deterministic": True,
    "workers": 8,
}


@pytest.fixture
def run_dir(tmp_path):
    (tmp_path / "args.yaml").write_text(yaml.safe_dump(RUN_ARGS))
    (tmp_path / "results.csv").write_text(RESULTS_CSV)
    return tmp_path


def test_results_csv_header_is_stripped(run_dir):
    rows = read_results_csv(run_dir / "results.csv")
    assert len(rows) == 3
    assert rows[1]["metrics/mAP50(B)"] == pytest.approx(0.97)
    assert rows[2]["epoch"] == 3


def test_summary_reports_final_and_best_validation_epochs(run_dir):
    summary = summarize_training_run(run_dir)
    assert summary["epochs_run"] == 3
    assert summary["metrics_split"] == "val"
    assert summary["final_epoch"]["epoch"] == 3
    assert summary["final_epoch"]["mAP50"] == pytest.approx(0.98)
    # mAP50-95 peaked one epoch before the end.
    assert summary["highest_mAP50-95_epoch"]["epoch"] == 2
    assert summary["highest_mAP50-95_epoch"]["dfl_loss"] == pytest.approx(1.1)
    assert summary["training_hours"] == pytest.approx(212.4 / 3600)
    assert "checkpoint" not in summary


def test_checkpoint_metadata_is_matched_to_its_epoch(run_dir):
    torch = pytest.importorskip("torch")
    weights = run_dir / "best.pt"
    torch.save(
        {
            "date": "2025-03-30T14:02:11",
            "version": "8.3.99",
            "best_fitness": 0.727,
            "train_args": {**RUN_ARGS, "workers": 4},
            "train_metrics": {"metrics/mAP50(B)": 0.97, "metrics/mAP50-95(B)": 0.70},
        },
        weights,
    )
    checkpoint = summarize_training_run(run_dir, weights)["checkpoint"]
    assert checkpoint["date"] == "2025-03-30T14:02:11"
    assert checkpoint["ultralytics_version"] == "8.3.99"
    assert checkpoint["epoch"] == 2
    assert checkpoint["metrics"] == {"mAP50": 0.97, "mAP50-95": 0.70}
    assert checkpoint["args_differ_from_args_yaml"] == [{"key": "workers", "run": 8, "config": 4}]


def test_empty_results_are_an_error(run_dir):
    (run_dir / "results.csv").write_text(",".join(COLUMNS) + "\n")
    with pytest.raises(ValueError, match="no epochs"):
        summarize_training_run(run_dir)


def test_config_differences_cover_config_and_key_settings():
    config = {"model": "weights/yolo11n.pt", "data": "data/plates/data.yaml", "epochs": 3}
    config.update(batch=16, imgsz=640, seed=0, deterministic=True, workers=4)
    differences = config_differences(RUN_ARGS, config)
    assert differences == [
        # The config leaves these to the Ultralytics default, which may since have changed.
        {"key": "lr0", "run": 0.01, "config": None},
        {"key": "momentum", "run": 0.937, "config": None},
        {"key": "workers", "run": 8, "config": 4},
    ]


def test_shipped_training_config_loads():
    args = load_train_args("configs/train_plate_detector.yaml", ["epochs=1"])
    assert args["epochs"] == 1
    assert args["model"] and args["data"]
