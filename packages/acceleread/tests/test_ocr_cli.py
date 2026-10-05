# SPDX-License-Identifier: Apache-2.0
"""`acceleread ocr add-language` and `acceleread doctor` (spec §2, §4.3, §9)."""

from pathlib import Path

import pytest

from acceleread.cli import main


def test_add_language_from_file_installs_into_the_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = tmp_path / "german.traineddata"
    pack.write_bytes(b"pack")
    ws = tmp_path / "ws"
    assert (
        main(["--workspace", str(ws), "ocr", "add-language", "de", "--from-file", str(pack)]) == 0
    )
    assert (ws / "models" / "tessdata" / "deu.traineddata").read_bytes() == b"pack"
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
    pack.write_bytes(b"pack")
    main(["--workspace", str(ws), "ocr", "add-language", "fr", "--from-file", str(pack)])
    capsys.readouterr()
    assert main(["--workspace", str(ws), "doctor"]) == 0
    out = capsys.readouterr().out
    assert "tesseract" in out and "5." in out
    assert "en" in out and "fr" in out  # vendored and Workspace packs
    assert "models" in out and "absent" in out  # Docling and PP-OCR weights, optional
    assert "quality" in out  # the extra
    assert "offline" in out


def test_doctor_does_not_create_the_workspace(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    assert main(["--workspace", str(ws), "doctor"]) == 0
    assert not ws.exists()


def test_doctor_fails_when_the_vendored_english_pack_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("acceleread.doctor.VENDORED_TESSDATA", tmp_path / "none")
    assert main(["--workspace", str(tmp_path / "ws"), "doctor"]) == 1
    assert "eng" in capsys.readouterr().out
