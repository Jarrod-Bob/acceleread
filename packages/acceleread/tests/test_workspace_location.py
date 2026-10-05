# SPDX-License-Identifier: Apache-2.0
"""Workspace location, layout and the network-filesystem refusal (spec §8)."""

import sqlite3
from pathlib import Path

import pytest

from acceleread.workspace import NetworkFilesystemError, Workspace, resolve_workspace_path


def test_explicit_path_beats_env_and_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("ACCELEREAD_HOME", str(tmp_path / "env"))
    assert resolve_workspace_path(tmp_path / "flag") == tmp_path / "flag"


def test_env_var_beats_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("ACCELEREAD_HOME", str(tmp_path / "env"))
    assert resolve_workspace_path(None) == tmp_path / "env"


def test_default_is_dot_acceleread_in_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("ACCELEREAD_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_workspace_path(None) == tmp_path / ".acceleread"


def test_opening_creates_the_layout(tmp_path: Path):
    with Workspace.open(tmp_path / "ws") as ws:
        assert ws.path == tmp_path / "ws"
    assert (tmp_path / "ws" / "catalog.sqlite").is_file()
    assert (tmp_path / "ws" / "cache.sqlite").is_file()
    assert (tmp_path / "ws" / "jobs").is_dir()
    assert (tmp_path / "ws" / "models").is_dir()


def test_catalog_and_cache_use_wal(tmp_path: Path):
    with Workspace.open(tmp_path):
        pass
    for name in ("catalog.sqlite", "cache.sqlite"):
        with sqlite3.connect(tmp_path / name) as db:
            assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)


def test_network_filesystem_is_refused(tmp_path: Path):
    with pytest.raises(NetworkFilesystemError, match="allow-network-fs"):
        Workspace.open(tmp_path, filesystem_type=lambda _path: "nfs4")


def test_network_filesystem_is_allowed_when_asked(tmp_path: Path):
    with Workspace.open(tmp_path, allow_network_fs=True, filesystem_type=lambda _p: "smbfs"):
        pass


def test_local_filesystem_is_accepted(tmp_path: Path):
    with Workspace.open(tmp_path, filesystem_type=lambda _path: "apfs"):
        pass


def test_refused_workspace_is_left_untouched(tmp_path: Path):
    with pytest.raises(NetworkFilesystemError):
        Workspace.open(tmp_path / "ws", filesystem_type=lambda _path: "cifs")
    assert not (tmp_path / "ws").exists()
