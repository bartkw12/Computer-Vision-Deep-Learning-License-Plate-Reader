import csv
import json

import cv2
import numpy as np
import pytest

import synthetic
from alpr import __version__
from alpr.cli import main
from alpr.io import VideoReader, VideoWriter
from alpr.pipeline import PlateReaderPipeline
from alpr.types import PlateRead

TRUTH = {v.plate for v in synthetic.VEHICLES}


@pytest.fixture(scope="module")
def scene_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("cli") / "scene.mp4"
    with VideoWriter(path, fps=25, size=(synthetic.WIDTH, synthetic.HEIGHT)) as writer:
        for _, frame in synthetic.frames():
            writer.write(frame)
    return path


@pytest.fixture
def stand_in_models(monkeypatch):
    """Make ``alpr run`` build its pipeline from the synthetic stand-ins."""
    monkeypatch.setattr(
        "alpr.pipeline.build_pipeline",
        lambda config: PlateReaderPipeline(*synthetic.stand_ins(), config),
    )


def run_cli(*args):
    return main(["--quiet", *map(str, args)])


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"alpr {__version__}"


def test_a_command_is_required(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2


def test_run_writes_a_complete_run_directory(scene_video, tmp_path, stand_in_models, capsys):
    out = tmp_path / "run"
    assert run_cli("run", scene_video, "--out", out, "--set", "tracker.max_age=5") == 0

    assert {p.name for p in out.iterdir()} == {
        "tracks.csv",
        "plates.csv",
        "timing.json",
        "run.json",
    }
    with (out / "plates.csv").open(newline="") as file:
        assert {row["text"] for row in csv.DictReader(file)} == TRUTH
    meta = json.loads((out / "run.json").read_text())
    assert meta["video"]["frames_processed"] == synthetic.N_FRAMES
    assert meta["video"]["width"] == synthetic.WIDTH
    assert meta["config"]["tracker"]["max_age"] == 5
    assert meta["counters"]["ocr_calls"] > 0
    assert meta["environment"]["packages"]["alpr"] == __version__
    timing = json.loads((out / "timing.json").read_text())
    assert timing["frames"] == synthetic.N_FRAMES and "ocr" in timing["stages"]

    printed = capsys.readouterr().out
    assert all(plate in printed for plate in TRUTH)
    assert f"{synthetic.N_FRAMES} frames" in printed


def test_run_can_stop_early_and_render(scene_video, tmp_path, stand_in_models):
    out = tmp_path / "run"
    assert run_cli("run", scene_video, "--out", out, "--max-frames", 30, "--render") == 0
    assert json.loads((out / "run.json").read_text())["video"]["frames_processed"] == 30
    with VideoReader(out / "annotated.mp4") as video:
        # The render covers the whole clip; only the analysis stopped early.
        assert len(list(video)) == synthetic.N_FRAMES


def test_full_workflow_run_render_label_evaluate(scene_video, tmp_path, stand_in_models, capsys):
    run_dir, labels, summary = tmp_path / "run", tmp_path / "gt" / "scene.csv", tmp_path / "s.json"
    assert run_cli("run", scene_video, "--out", run_dir, "--set", "tracker.max_age=5") == 0

    assert run_cli("render", run_dir, "--no-insets") == 0
    assert (run_dir / "annotated.mp4").stat().st_size > 1000

    assert run_cli("label-sheet", run_dir, "--out", labels) == 0
    assert (tmp_path / "gt" / "scene_sheet_01.jpg").exists()
    assert run_cli("label-sheet", run_dir, "--out", labels) == 2, "refuses to overwrite"
    assert "already exists" in capsys.readouterr().err

    # Type in the answers.
    with labels.open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        frame, x, y = int(row["frame"]), float(row["x"]), float(row["y"])
        row["plate"] = next(
            v.plate
            for v in synthetic.VEHICLES
            if v.box(frame) and v.box(frame).contains_point(x, y)
        )
    with labels.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    assert run_cli("eval-video", run_dir, "--labels", labels, "--out", summary) == 0
    assert "3/3 labeled vehicles read correctly" in capsys.readouterr().out

    published = summary.read_text()
    report = json.loads(published)
    assert report["summary"]["reading"]["accuracy"] == 1.0
    assert report["config"]["plate_format"]["name"] == "ontario"
    # What gets committed must not identify the footage or the vehicles in it.
    assert not any(plate in published for plate in TRUTH)
    assert "path" not in report["video"] and str(scene_video) not in published
    records = json.loads((run_dir / "eval_records.json").read_text())
    assert {record["truth"] for record in records} == TRUTH


def test_missing_plate_weights_are_explained(scene_video, tmp_path, capsys):
    code = run_cli("run", scene_video, "--set", f"plate.weights={tmp_path / 'absent.pt'}")
    assert code == 2
    error = capsys.readouterr().err
    assert "plate detector weights not found" in error and "alpr train" in error


def test_missing_video_and_bad_settings_exit_cleanly(tmp_path, stand_in_models, capsys):
    assert run_cli("run", tmp_path / "nope.mp4", "--out", tmp_path / "run") == 2
    assert "video not found" in capsys.readouterr().err
    assert run_cli("run", tmp_path / "nope.mp4", "--set", "tracker.max_agee=3") == 2
    assert "unknown config key" in capsys.readouterr().err
    assert run_cli("render", tmp_path / "not_a_run") == 2
    assert "not a run directory" in capsys.readouterr().err


def test_eval_images_writes_a_report(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "bench" / "endtoend" / "us"
    folder.mkdir(parents=True)
    for name, text in [("a", "ABC123"), ("b", "XYZ789"), ("c", "JKL456")]:
        cv2.imwrite(str(folder / f"{name}.jpg"), np.full((60, 120, 3), 128, np.uint8))
        (folder / f"{name}.txt").write_text(f"{name}.jpg\t10\t10\t80\t30\t{text}\n")

    class AlwaysAbc:
        name, preprocess_name = "fake", "none"

        def read(self, crop):
            return PlateRead("ABC123", 0.9)

    monkeypatch.setattr("alpr.ocr.build_ocr", lambda *args, **kwargs: AlwaysAbc())
    out = tmp_path / "results" / "stills.json"
    code = run_cli(
        "eval-images", "--data", tmp_path / "bench", "--regions", "us", "--device", "cpu",
        "--limit", 2, "--out", out,
    )  # fmt: skip
    assert code == 0
    report = json.loads(out.read_text())
    assert report["recognition"]["overall"]["n"] == 2
    assert report["recognition"]["overall"]["exact"] == 1
    assert "end_to_end" not in report
    assert report["environment"]["device"] == "cpu"
    assert "1/2 exact" in capsys.readouterr().out
