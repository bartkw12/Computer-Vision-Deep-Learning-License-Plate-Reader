import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "render_readme_tables.py"
_spec = importlib.util.spec_from_file_location("render_readme_tables", SCRIPT)
render = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = render  # dataclasses look their module up here
_spec.loader.exec_module(render)


def score(accuracy, n=10):
    return {"n": n, "exact": round(accuracy * n), "accuracy": accuracy, "cer": 0.1,
            "ci_low": accuracy / 2, "ci_high": (1 + accuracy) / 2}  # fmt: skip


def variant(ocr, preprocess, accuracy):
    return {
        "ocr": ocr,
        "preprocess": preprocess,
        "ms_per_plate": 12.3,
        "overall": score(accuracy, 20),
        "by_region": {"br": score(accuracy), "us": score(accuracy / 2)},
        "by_plate_height": {"0-20px": score(accuracy / 3, 4), "70px+": score(accuracy, 16)},
        "top_confusions": [{"truth": "0", "predicted": "O", "count": 3}],
        "format_correction_br": [
            {"mode": "off", "emitted": 10, **score(accuracy)},
            {"mode": "lookalike-2", "emitted": 9, **score(min(1.0, accuracy + 0.1))},
        ],
    }


STILLS = {
    "plates": 20,
    "variants": [
        variant("easyocr", "none", 0.25),
        variant("easyocr", "otsu_inv", 0.3),
        variant("fast-plate-ocr:cct-s-v2-global-model", "none", 0.9),
    ],
    "environment": {"cpu": "Test CPU", "device": "cpu"},
}

README = """\
# Title

<!-- results:headline -->
stale text
<!-- /results:headline -->

Prose that is left alone.

<!-- results:ocr-stills -->
<!-- /results:ocr-stills -->

<!-- results:detector -->
old 0.985
<!-- /results:detector -->
"""


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    readme = tmp_path / "README.md"
    readme.write_text(README)
    return readme, results


def run(readme, results, *extra):
    return render.main(["--readme", str(readme), "--results", str(results), *extra])


def test_missing_results_render_as_pending_not_as_old_numbers(project):
    readme, results = project
    assert run(readme, results, "--check") == 1
    assert run(readme, results) == 0
    text = readme.read_text()
    assert "0.985" not in text and "stale text" not in text
    assert text.count("> _Pending: produced by") == 3
    assert "Prose that is left alone." in text
    assert run(readme, results, "--check") == 0


def test_blocks_render_from_results(project):
    readme, results = project
    (results / render.STILLS).write_text(json.dumps(STILLS))
    (results / "detector_original_test.json").write_text(
        json.dumps(
            {
                "images": 50,
                "instances": 61,
                "precision": 0.91,
                "recall": 0.88,
                "mAP50": 0.93,
                "mAP50-95": 0.61,
            }
        )
    )
    run(readme, results)
    text = readme.read_text()
    assert "reads **90.0% (45-95)** exactly" in text
    assert "against 30.0% (15-65) for the first version's EasyOCR + Otsu" in text
    assert "mAP50 **0.930**" in text
    assert "| `easyocr` | `otsu_inv` (v1) | 30.0% (15-65) |" in text
    assert "| original, **test** | 50 | 61 | 0.910 | 0.880 | 0.930 | 0.610 |" in text
    # Only the test split exists, so no validation row is invented.
    assert "original, val" not in text
    assert run(readme, results, "--check") == 0

    # A changed results file makes the README stale again.
    changed = {**STILLS, "variants": [*STILLS["variants"][:2], variant("x", "none", 0.95)]}
    (results / render.STILLS).write_text(json.dumps(changed))
    assert run(readme, results, "--check") == 1


@pytest.mark.parametrize("name", ["format-br", "plate-height", "confusions"])
def test_stills_side_tables_render(tmp_path, name):
    (tmp_path / render.STILLS).write_text(json.dumps(STILLS))
    text = render.render_block(name, tmp_path)
    assert "EasyOCR + Otsu (v1)" in text
    assert "fast-plate-ocr" in text


def test_leakage_and_training_blocks_render(tmp_path):
    (tmp_path / render.LEAKAGE).write_text(
        json.dumps(
            {
                "thresholds": [0, 4],
                "split_sizes": {"train": 6, "valid": 3, "test": 0},
                "pairs": {
                    "valid_vs_train": {
                        "queries": 3,
                        "within": {
                            "0": {"count": 1, "share": 1 / 3},
                            "4": {"count": 2, "share": 2 / 3},
                        },
                    },
                    "test_vs_train": {"queries": 0, "within": {}},
                },
            }
        )
    )
    leakage = render.render_block("leakage", tmp_path)
    assert "| valid vs train | 3 | 1 (33.3%) | 2 (66.7%) |" in leakage
    assert "| test vs train | 0 | - | - |" in leakage

    (tmp_path / render.TRAINING).write_text(
        json.dumps(
            {
                "args": {"model": "yolo11n.pt", "epochs": 100, "lr0": 0.01},
                "epochs_run": 100,
                "training_hours": 2.04,
                "checkpoint": {"ultralytics_version": "8.3.99", "date": "2025-03-30T14:02:11"},
                "config_differences": [{"key": "workers", "run": 8, "config": 4}],
            }
        )
    )
    training = render.render_block("training-original", tmp_path)
    assert (
        "100 epochs, 2.0 h of training, Ultralytics 8.3.99, checkpoint saved 2025-03-30" in training
    )
    assert "| `lr0` | `0.01` |" in training
    assert "`workers` (run `8`, config `4`)" in training


def test_unknown_block_is_an_error(project):
    readme, results = project
    readme.write_text("<!-- results:nonsense -->\n<!-- /results:nonsense -->\n")
    with pytest.raises(SystemExit, match="nothing renders"):
        run(readme, results)


def test_results_without_a_renderer_are_an_error(tmp_path):
    (tmp_path / "video_eval.json").write_text("{}")
    with pytest.raises(SystemExit, match="no renderer"):
        render.render_block("video", tmp_path)


def test_every_registered_block_is_in_the_readme():
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    assert {match["name"] for match in render.BLOCK.finditer(readme)} == set(render.BLOCKS)
