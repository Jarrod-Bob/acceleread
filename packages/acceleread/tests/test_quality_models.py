# SPDX-License-Identifier: Apache-2.0
"""`acceleread models fetch` and the model layout the `quality` Profile reads (spec §2, §4.1).

The downloader is the seam: tests hand `fetch_models` a fake that writes known bytes, so nothing
touches the network.
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

from acceleread import quality_models
from acceleread.cli import main
from acceleread.doctor import run_checks
from acceleread.quality_models import (
    ModelFetchError,
    ModelPin,
    check_model,
    fetch_models,
    models_dir,
    models_status,
)

LAYOUT_BYTES = b"layout weights"
OCR_BYTES = b"ocr weights"
TABLE_BYTES = b"table weights"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pins(layout: bytes = LAYOUT_BYTES, ocr: bytes = OCR_BYTES) -> tuple[ModelPin, ...]:
    return (
        ModelPin(
            name="layout",
            label="Docling layout",
            source="hf",
            folder="org--layout",
            repo_id="org/layout",
            revision="abc123",
            files={"model.bin": sha(layout)},
        ),
        ModelPin(
            name="ocr",
            label="RapidOCR (PP-OCR latin)",
            source="rapidocr",
            folder="RapidOcr",
            revision="rapidocr-9",
            files={"rec.onnx": sha(ocr)},
        ),
        ModelPin(
            name="tables",
            label="TableFormer (optional)",
            source="hf",
            folder="org--tables",
            repo_id="org/tables",
            revision="v1",
            files={"t.bin": sha(TABLE_BYTES)},
            optional=True,
        ),
    )


PINS = pins()


class FakeDownloader:
    def __init__(self, layout: bytes = LAYOUT_BYTES, ocr: bytes = OCR_BYTES) -> None:
        self.layout = layout
        self.ocr = ocr
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
        (local_dir / "rec.onnx").write_bytes(self.ocr)


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
    assert sorted(p.name for p in root.iterdir()) == ["RapidOcr", "manifest.json", "org--layout"]


def test_fetch_with_tables_adds_tableformer(tmp_path: Path) -> None:
    dl = FakeDownloader()
    manifest = fetch_models(tmp_path, tables=True, pins=PINS, downloader=dl)
    assert "hf:org/tables@v1" in dl.calls
    assert manifest["tables"]["files"] == {"org--tables/t.bin": sha(TABLE_BYTES)}


def test_adding_tables_later_keeps_the_verified_weights_and_downloads_only_tableformer(
    tmp_path: Path,
) -> None:
    fetch_models(tmp_path, pins=PINS, downloader=FakeDownloader())
    later = FakeDownloader()
    manifest = fetch_models(tmp_path, tables=True, pins=PINS, downloader=later)
    assert later.calls == ["hf:org/tables@v1"]
    assert set(manifest) == {"layout", "ocr", "tables"}
    # A later fetch without --tables doesn't forget the installed TableFormer.
    assert "tables" in fetch_models(tmp_path, pins=PINS, downloader=FakeDownloader())


def test_a_file_that_does_not_match_its_pinned_hash_fails_the_fetch(tmp_path: Path) -> None:
    with pytest.raises(ModelFetchError, match=r"model\.bin"):
        fetch_models(tmp_path, pins=PINS, downloader=FakeDownloader(layout=b"tampered"))
    root = models_dir(tmp_path)
    assert not (root / "manifest.json").exists()
    assert not (root / "org--layout").exists()
    assert list(root.iterdir()) == []  # no temp directories left behind


def test_a_failed_fetch_leaves_the_previous_install_and_manifest_untouched(
    tmp_path: Path,
) -> None:
    fetch_models(tmp_path, pins=PINS, downloader=FakeDownloader())
    root = models_dir(tmp_path)
    before = (root / "manifest.json").read_text()

    # A new pin set: the layout weights changed (verifies), but the OCR download is bad.
    newer = pins(layout=b"layout v2", ocr=b"ocr v2")
    with pytest.raises(ModelFetchError, match=r"rec\.onnx"):
        fetch_models(
            tmp_path, pins=newer, downloader=FakeDownloader(layout=b"layout v2", ocr=b"bad")
        )

    assert (root / "org--layout" / "model.bin").read_bytes() == LAYOUT_BYTES  # not replaced
    assert (root / "RapidOcr" / "rec.onnx").read_bytes() == OCR_BYTES
    assert (root / "manifest.json").read_text() == before
    assert sorted(p.name for p in root.iterdir()) == ["RapidOcr", "manifest.json", "org--layout"]


def test_a_changed_pin_replaces_the_old_weights_atomically(tmp_path: Path) -> None:
    fetch_models(tmp_path, pins=PINS, downloader=FakeDownloader())
    fetch_models(tmp_path, pins=pins(b"layout v2"), downloader=FakeDownloader(layout=b"layout v2"))
    root = models_dir(tmp_path)
    assert (root / "org--layout" / "model.bin").read_bytes() == b"layout v2"
    assert sorted(p.name for p in root.iterdir()) == ["RapidOcr", "manifest.json", "org--layout"]


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


def test_check_model_tells_absent_from_corrupt_from_ok(tmp_path: Path) -> None:
    layout = PINS[0]
    assert check_model(tmp_path, layout) == "absent"
    fetch_models(tmp_path, pins=PINS, downloader=FakeDownloader())
    assert check_model(tmp_path, layout) == "ok"
    weights = models_dir(tmp_path) / "org--layout" / "model.bin"
    weights.write_bytes(LAYOUT_BYTES[:4])  # truncated
    assert check_model(tmp_path, layout) == "corrupt"
    weights.unlink()
    assert check_model(tmp_path, layout) == "absent"


def test_status_means_verified_not_merely_present(tmp_path: Path) -> None:
    assert models_status(tmp_path, pins=PINS) == {"layout": False, "ocr": False, "tables": False}
    fetch_models(tmp_path, pins=PINS, downloader=FakeDownloader())
    assert models_status(tmp_path, pins=PINS) == {"layout": True, "ocr": True, "tables": False}
    (models_dir(tmp_path) / "RapidOcr" / "rec.onnx").write_bytes(b"tampered!")
    assert models_status(tmp_path, pins=PINS)["ocr"] is False


def test_verification_is_cached_until_the_files_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch_models(tmp_path, pins=PINS, downloader=FakeDownloader())
    hashed: list[Path] = []
    real = quality_models._sha256
    monkeypatch.setattr(quality_models, "_sha256", lambda p: (hashed.append(p), real(p))[1])
    assert check_model(tmp_path, PINS[0]) == "ok"
    first = len(hashed)
    assert first >= 1
    assert check_model(tmp_path, PINS[0]) == "ok"
    assert len(hashed) == first  # not re-hashed for the next Document
    (models_dir(tmp_path) / "org--layout" / "model.bin").write_bytes(b"other")
    assert check_model(tmp_path, PINS[0]) == "corrupt"


def test_every_real_pin_names_a_commit_and_hashes_every_file() -> None:
    by_name = {p.name: p for p in quality_models.PINS}
    assert set(by_name) == {"layout", "ocr", "tables"}
    for pin in by_name.values():
        assert pin.files, pin.name
        assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in pin.files.values()), pin.name
    for name in ("layout", "tables"):
        assert re.fullmatch(r"[0-9a-f]{40}", by_name[name].revision), name  # a commit, never a tag
    assert by_name["tables"].optional and not by_name["layout"].optional


def test_a_hugging_face_pin_needs_a_repository() -> None:
    with pytest.raises(ValueError, match="repo_id"):
        ModelPin(name="x", source="hf", folder="x", revision="r", files={"a": "0" * 64})


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
    (models_dir(ws) / "RapidOcr" / "rec.onnx").write_bytes(b"trunc")
    rows = {c.name: c for c in run_checks(ws)}
    assert rows["models: RapidOCR (PP-OCR latin)"].status == "error"
    assert "models fetch" in rows["models: RapidOCR (PP-OCR latin)"].detail
