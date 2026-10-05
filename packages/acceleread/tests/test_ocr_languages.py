# SPDX-License-Identifier: Apache-2.0
"""OCR languages (spec §4.3): the ISO 639-1 table, installed packs and `ocr add-language`."""

from pathlib import Path

import pytest

from acceleread.extract import VENDORED_TESSDATA
from acceleread.languages import (
    LanguagePackError,
    add_language,
    installed_languages,
    rapidocr_code,
    tesseract_code,
    workspace_tessdata,
)


def test_iso_codes_map_to_tesseract_and_rapidocr_codes() -> None:
    assert tesseract_code("en") == "eng"
    assert tesseract_code("de") == "deu"
    assert tesseract_code("zh") == "chi_sim"
    assert rapidocr_code("de") == "latin"
    assert rapidocr_code("ja") is None  # no baked RapidOCR family for it in v0
    with pytest.raises(KeyError):
        tesseract_code("xx")


def test_only_the_vendored_english_pack_is_installed_by_default() -> None:
    assert installed_languages([VENDORED_TESSDATA]) == {"en"}


def test_add_language_from_a_file_installs_a_pack_the_workspace_then_reports(
    tmp_path: Path,
) -> None:
    source = tmp_path / "anything.traineddata"
    source.write_bytes(b"pack bytes")
    tessdata = workspace_tessdata(tmp_path / "ws")
    path = add_language("de", tessdata, from_file=source)
    assert path == tessdata / "deu.traineddata"
    assert path.read_bytes() == b"pack bytes"
    assert installed_languages([VENDORED_TESSDATA, tessdata]) == {"en", "de"}


def test_add_language_downloads_through_the_fetcher(tmp_path: Path) -> None:
    asked: list[str] = []

    def fetch(pack: str) -> bytes:
        asked.append(pack)
        return b"downloaded"

    path = add_language("fr", tmp_path / "tessdata", fetch=fetch)
    assert asked == ["fra"]
    assert path.read_bytes() == b"downloaded"


def test_add_language_rejects_unknown_codes_and_auto(tmp_path: Path) -> None:
    for code in ("xx", "auto"):
        with pytest.raises(LanguagePackError, match=code):
            add_language(code, tmp_path, fetch=lambda pack: b"x")
    assert not list(tmp_path.iterdir())


def test_add_language_is_an_error_when_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACCELEREAD_OFFLINE", "1")

    def fetch(pack: str) -> bytes:
        raise AssertionError("must not download")

    with pytest.raises(LanguagePackError, match="ACCELEREAD_OFFLINE"):
        add_language("de", tmp_path, fetch=fetch)


def test_add_language_rejects_an_empty_pack(tmp_path: Path) -> None:
    with pytest.raises(LanguagePackError, match="empty"):
        add_language("de", tmp_path, fetch=lambda pack: b"")
    assert not (tmp_path / "deu.traineddata").exists()
