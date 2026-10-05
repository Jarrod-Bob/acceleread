# SPDX-License-Identifier: Apache-2.0
"""`acceleread models fetch` and the model layout the `quality` Profile reads (spec §2, §4.1).

The downloader is the seam: tests hand `fetch_models` a fake that writes known bytes, so nothing
touches the network.
"""

import hashlib
import json
from pathlib import Path

import pytest

from acceleread import quality_models
from acceleread.cli import main
from acceleread.doctor import run_checks
from acceleread.quality_models import (
    ModelFetchError,
    ModelPin,
    fetch_models,
    models_dir,
    models_status,
)

LAYOUT_BYTES = b"layout weights"
OCR_BYTES = b"ocr weights"
TABLE_BYTES = b"table weights"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


PINS = (
    ModelPin(
        name="layout",
        label="Docling layout",
        source="hf",
        folder="org--layout",
        repo_id="org/layout",
        revision="abc123",
        files={"model.bin": sha(LAYOUT_BYTES)},
    ),
    ModelPin(
        name="ocr",
        label="RapidOCR (PP-OCR latin)",
        source="rapidocr",
        folder="RapidOcr",
        revision="rapidocr-9",
        files={"rec.onnx": sha(OCR_BYTES)},
    ),
    ModelPin(
        name="tables",
        label="TableFormer (optional)",
        source="hf",
        folder="org--tables",
        repo_id="org/tables",
        revision="v1",
        files={},
        optional=True,
    ),
)


class FakeDownloader:
    def __init__(self, layout: bytes = LAYOUT_BYTES) -> None:
        self.layout = layout
        self.calls: list[str] = []

    def hf(self, repo_id: str, revision: str, local_dir: Path, files: list[str]) -> None:
        self.calls.append(f"hf:{repo_id}@{revision}")
        local_dir.mkdir(parents=True, exist_ok=True)
        if repo_id == "org/layout":
            (local_dir / "model.bin").write_bytes(self.layout)
        else:
            (local_dir / "t.bin").write_bytes(TABLE_BYTES)

    def rapidocr(self, local_dir: Path) -> None:
        self.calls.append("rapidocr")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "rec.onnx").write_bytes(OCR_BYTES)


def test_fetch_downloads_the_pinned_models_into_the_workspace_and_records_hashes(
    tmp_path: Path,
) -> None:
    dl = FakeDownloader()
    manifest = fetch_models(tmp_path, pins=PINS, downloader=dl)

    root = models_dir(tmp_path)
    assert root == tmp_path / "models" / "docling"
    assert (root / "org--layout" / "model.bin").read_bytes() == LAYOUT_BYTES
    assert (root / "RapidOcr" / "rec.onnx").read_bytes() == OCR_BYTES
    assert dl.calls == ["hf:org/layout@abc123", "rapidocr"]  # tables are opt-in

    saved = json.loads((root / "manifest.json").read_text())
    assert saved == manifest
    assert saved["layout"] == {
        "revision": "abc123",
        "files": {"org--layout/model.bin": sha(LAYOUT_BYTES)},
    }
    assert "tables" not in saved


def test_fetch_with_tables_adds_tableformer(tmp_path: Path) -> None:
    dl = FakeDownloader()
    manifest = fetch_models(tmp_path, tables=True, pins=PINS, downloader=dl)
    assert "hf:org/tables@v1" in dl.calls
    assert manifest["tables"]["files"] == {"org--tables/t.bin": sha(TABLE_BYTES)}


def test_a_file_that_does_not_match_its_pinned_hash_fails_the_fetch(tmp_path: Path) -> None:
    with pytest.raises(ModelFetchError, match=r"model\.bin"):
        fetch_models(tmp_path, pins=PINS, downloader=FakeDownloader(layout=b"tampered"))
    assert not (models_dir(tmp_path) / "manifest.json").exists()
    assert not (models_dir(tmp_path) / "org--layout" / "model.bin").exists()


def test_fetch_again_downloads_nothing_that_is_already_verified(tmp_path: Path) -> None:
    fetch_models(tmp_path, pins=PINS, downloader=FakeDownloader())
    again = FakeDownloader()
    fetch_models(tmp_path, pins=PINS, downloader=again)
    assert again.calls == []


def test_offline_mode_forbids_fetching(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCELEREAD_OFFLINE", "1")
    dl = FakeDownloader()
    with pytest.raises(ModelFetchError, match="ACCELEREAD_OFFLINE"):
        fetch_models(tmp_path, pins=PINS, downloader=dl)
    assert dl.calls == []


def test_status_reports_each_model_as_installed_or_absent(tmp_path: Path) -> None:
    assert models_status(tmp_path, pins=PINS) == {"layout": False, "ocr": False, "tables": False}
    fetch_models(tmp_path, pins=PINS, downloader=FakeDownloader())
    assert models_status(tmp_path, pins=PINS) == {"layout": True, "ocr": True, "tables": False}
    (models_dir(tmp_path) / "RapidOcr" / "rec.onnx").unlink()
    assert models_status(tmp_path, pins=PINS)["ocr"] is False


def test_the_real_pins_name_revisions_and_hashes() -> None:
    by_name = {p.name: p for p in quality_models.PINS}
    assert set(by_name) == {"layout", "ocr", "tables"}
    layout = by_name["layout"]
    assert len(layout.revision) == 40  # a commit, never `main`
    assert all(len(h or "") == 64 for h in layout.files.values())
    assert by_name["ocr"].files and by_name["tables"].optional


def test_models_fetch_command_reports_what_it_installed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(quality_models, "PINS", PINS)
    monkeypatch.setattr(quality_models, "default_downloader", FakeDownloader)
    ws = tmp_path / "ws"
    assert main(["--workspace", str(ws), "models", "fetch"]) == 0
    out = capsys.readouterr().out
    assert "layout" in out and "ocr" in out and "tables" not in out
    assert (ws / "models" / "docling" / "manifest.json").is_file()


def test_models_fetch_command_fails_cleanly_offline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACCELEREAD_OFFLINE", "1")
    assert main(["--workspace", str(tmp_path / "ws"), "models", "fetch"]) == 1
    assert "ACCELEREAD_OFFLINE" in capsys.readouterr().err


def test_doctor_reports_the_model_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quality_models, "PINS", PINS)
    ws = tmp_path / "ws"
    rows = {c.name: c.status for c in run_checks(ws)}
    assert rows["models: Docling layout"] == "absent"
    assert rows["models: RapidOCR (PP-OCR latin)"] == "absent"
    assert rows["models: TableFormer (optional)"] == "absent"
    fetch_models(ws, pins=PINS, downloader=FakeDownloader())
    rows = {c.name: c.status for c in run_checks(ws)}
    assert rows["models: Docling layout"] == "ok"
    assert rows["models: RapidOCR (PP-OCR latin)"] == "ok"
