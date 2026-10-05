# SPDX-License-Identifier: Apache-2.0
"""`acceleread ocr add-language` and `acceleread doctor` (spec §2, §4.3, §9)."""

from pathlib import Path

import pytest

from acceleread.cli import main
from acceleread.doctor import run_checks
from acceleread.extract import VENDORED_TESSDATA

ENG_PACK = (VENDORED_TESSDATA / "eng.traineddata").read_bytes()


def rows(out: str) -> dict[str, tuple[str, str]]:
    """Doctor's lines as {check name: (status, detail)}."""
    parsed = {}
    for line in out.splitlines():
        status, rest = line.split(None, 1)
        name, _, detail = rest.partition("  ")
        parsed[name.strip()] = (status, detail.strip())
    return parsed


def test_add_language_from_file_installs_into_the_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = tmp_path / "german.traineddata"
    pack.write_bytes(ENG_PACK)
    ws = tmp_path / "ws"
    assert (
        main(["--workspace", str(ws), "ocr", "add-language", "de", "--from-file", str(pack)]) == 0
    )
    assert (ws / "models" / "tessdata" / "deu.traineddata").read_bytes() == ENG_PACK
    assert "deu" in capsys.readouterr().out


def test_add_language_reports_an_unknown_language(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--workspace", str(tmp_path / "ws"), "ocr", "add-language", "xx"]) == 1
    assert "unknown language 'xx'" in capsys.readouterr().err


def test_add_language_offline_without_a_file_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACCELEREAD_OFFLINE", "1")
    assert main(["--workspace", str(tmp_path / "ws"), "ocr", "add-language", "de"]) == 1
    assert "ACCELEREAD_OFFLINE" in capsys.readouterr().err


def test_doctor_reports_tesseract_packs_models_and_extras(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    pack = tmp_path / "p"
    pack.write_bytes(ENG_PACK)
    main(["--workspace", str(ws), "ocr", "add-language", "fr", "--from-file", str(pack)])
    capsys.readouterr()
    assert main(["--workspace", str(ws), "doctor"]) == 0
    report = rows(capsys.readouterr().out)
    assert report["tesseract"][0] == "ok" and report["tesseract"][1]
    assert report["language packs (vendored)"] == ("ok", "en")
    assert report["language packs (workspace)"] == ("ok", "fr")
    assert report["models: Docling weights"][0] == "absent"
    assert report["models: PP-OCR weights"][0] == "absent"
    for extra in ("quality", "llm", "edgar"):
        assert report[f"extra [{extra}]"][0] in ("ok", "absent")
    assert report["offline mode"][0] == "ok"


def test_doctor_does_not_create_the_workspace(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    assert main(["--workspace", str(ws), "doctor"]) == 0
    assert not ws.exists()


def test_doctor_flags_a_missing_vendored_english_pack(tmp_path: Path) -> None:
    checks = run_checks(tmp_path / "ws", vendored=tmp_path / "none")
    (vendored,) = [c for c in checks if c.name == "language packs (vendored)"]
    assert vendored.status == "error" and "eng" in vendored.detail
