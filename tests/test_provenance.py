import shutil
import subprocess

import pytest

from alpr.provenance import environment_info, git_dirty

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture
def repo(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "code.py").write_text("x = 1\n")
    git("add", "code.py")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    return tmp_path


def test_clean_checkout_is_not_dirty(repo):
    assert git_dirty(repo) is False


def test_generated_results_do_not_count(repo):
    (repo / "results").mkdir()
    (repo / "results" / "new.json").write_text("{}")
    (repo / "assets").mkdir()
    (repo / "assets" / "figure.png").write_bytes(b"png")
    assert git_dirty(repo) is False


@pytest.mark.parametrize("change", ["edit", "new file"])
def test_code_changes_count(repo, change):
    if change == "edit":
        (repo / "code.py").write_text("x = 2\n")
    else:
        (repo / "other.py").write_text("")
    assert git_dirty(repo) is True


def test_checked_from_a_subfolder_the_whole_repo_counts(repo):
    # The package calls git from src/alpr/, so paths must resolve from the top.
    subfolder = repo / "src"
    subfolder.mkdir()
    (repo / "results").mkdir()
    (repo / "results" / "new.json").write_text("{}")
    assert git_dirty(subfolder) is False
    (repo / "code.py").write_text("x = 3\n")
    assert git_dirty(subfolder) is True


def test_outside_a_repository(tmp_path):
    assert git_dirty(tmp_path) is None


def test_environment_info_records_the_revision():
    info = environment_info("cpu")
    assert info["device"] == "cpu"
    assert set(info) >= {"git_commit", "git_dirty", "python", "packages", "cpu"}
