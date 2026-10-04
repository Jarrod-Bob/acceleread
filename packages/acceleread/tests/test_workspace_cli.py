# SPDX-License-Identifier: Apache-2.0
"""CLI housekeeping over a Workspace: jobs delete/prune, cache prune/clear (spec §8, §9)."""

from pathlib import Path

import pytest

from acceleread.cli import main, parse_duration
from acceleread.workspace import Workspace


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("90s", 90), ("15m", 900), ("12h", 43200), ("30d", 2592000), ("2w", 1209600)],
)
def test_parse_duration(text: str, seconds: int):
    assert parse_duration(text).total_seconds() == seconds


def test_parse_duration_rejects_garbage():
    with pytest.raises(ValueError, match="duration"):
        parse_duration("soon")


def test_workspace_flag_selects_the_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    with Workspace.open(tmp_path / "ws") as ws:
        job = ws.create_job({})
        ws.finish_job(job.id, "done")
    assert main(["--workspace", str(tmp_path / "ws"), "jobs", "delete", job.id]) == 0
    assert not (tmp_path / "ws" / "jobs" / job.id).exists()
    assert job.id in capsys.readouterr().out


def test_env_var_selects_the_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ACCELEREAD_HOME", str(tmp_path / "home"))
    with Workspace.open() as ws:
        job = ws.create_job({})
    assert main(["jobs", "delete", job.id]) == 0


def test_delete_of_a_running_job_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    with Workspace.open(tmp_path) as ws:
        job = ws.create_job({})
        ws.set_job_state(job.id, "running")
    assert main(["--workspace", str(tmp_path), "jobs", "delete", job.id]) == 1
    assert "running" in capsys.readouterr().err
    assert (tmp_path / "jobs" / job.id).exists()


def test_delete_of_unknown_job_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["--workspace", str(tmp_path), "jobs", "delete", "nope"]) == 1
    assert "nope" in capsys.readouterr().err


def test_jobs_prune_reports_what_it_pruned(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    with Workspace.open(tmp_path, clock=lambda: 1.0) as ws:
        job = ws.create_job({}, kind="ingest")
        ws.finish_job(job.id, "done")
    code = main(
        ["--workspace", str(tmp_path), "jobs", "prune", "--older-than", "1d", "--kind", "ingest"]
    )
    assert code == 0
    assert job.id in capsys.readouterr().out
    assert (tmp_path / "jobs" / job.id / "job.sqlite").exists()  # pruned, not deleted
    with Workspace.open(tmp_path) as ws:
        assert ws.get_job(job.id).pruned


def test_jobs_prune_keep_records(tmp_path: Path):
    with Workspace.open(tmp_path, clock=lambda: 1.0) as ws:
        job = ws.create_job({})
        ws.finish_job(job.id, "done")
    args = ["--workspace", str(tmp_path), "jobs", "prune", "--older-than", "1d", "--keep-records"]
    assert main(args) == 0
    assert (tmp_path / "jobs" / job.id / "job.sqlite").exists()
    with Workspace.open(tmp_path) as ws:
        assert ws.get_job(job.id).pruned


def test_cache_clear_and_prune(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    with Workspace.open(tmp_path, clock=lambda: 1.0) as ws:
        ws.cache.put("a", {})
        ws.cache.put("b", {})
    assert main(["--workspace", str(tmp_path), "cache", "prune", "--older-than", "1d"]) == 0
    assert "2" in capsys.readouterr().out
    with Workspace.open(tmp_path) as ws:
        ws.cache.put("c", {})
    assert main(["--workspace", str(tmp_path), "cache", "clear"]) == 0
    with Workspace.open(tmp_path) as ws:
        assert ws.cache.get("c") is None


def test_network_filesystem_is_refused_by_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr("acceleread.workspace.detect_filesystem_type", lambda _p: "nfs")
    assert main(["--workspace", str(tmp_path / "ws"), "cache", "clear"]) == 1
    assert "allow-network-fs" in capsys.readouterr().err
    assert main(["--workspace", str(tmp_path / "ws"), "--allow-network-fs", "cache", "clear"]) == 0
