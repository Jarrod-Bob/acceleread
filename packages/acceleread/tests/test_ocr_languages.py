# SPDX-License-Identifier: Apache-2.0
"""OCR languages (spec §4.3): the ISO 639-1 table, installed packs and `ocr add-language`."""

from pathlib import Path

import pytest

from acceleread.extract import VENDORED_TESSDATA
from acceleread.languages import (
    RAPIDOCR_CODES,
    TESSERACT_CODES,
    LanguagePackError,
    add_language,
    installed_languages,
    workspace_tessdata,
)

ENG_PACK = (VENDORED_TESSDATA / "eng.traineddata").read_bytes()  # a real pack, which loads


def test_iso_codes_map_to_tesseract_and_rapidocr_codes() -> None:
    assert TESSERACT_CODES["en"] == "eng"
    assert TESSERACT_CODES["de"] == "deu"
    assert TESSERACT_CODES["zh"] == "chi_sim"
    assert RAPIDOCR_CODES["de"] == "latin"
    assert "ja" not in RAPIDOCR_CODES  # no baked RapidOCR family for it in v0
    assert "xx" not in TESSERACT_CODES


def test_only_the_vendored_english_pack_is_installed_by_default() -> None:
    assert installed_languages([VENDORED_TESSDATA]) == {"en"}


def test_add_language_from_a_file_installs_a_pack_the_workspace_then_reports(
    tmp_path: Path,
) -> None:
    source = tmp_path / "anything.traineddata"
    source.write_bytes(ENG_PACK)
    tessdata = workspace_tessdata(tmp_path / "ws")
    path = add_language("de", tessdata, from_file=source)
    assert path == tessdata / "deu.traineddata"
    assert path.read_bytes() == ENG_PACK
    assert installed_languages([VENDORED_TESSDATA, tessdata]) == {"en", "de"}


def test_add_language_downloads_through_the_fetcher(tmp_path: Path) -> None:
    asked: list[str] = []

    def fetch(pack: str) -> bytes:
        asked.append(pack)
        return ENG_PACK

    path = add_language("fr", tmp_path / "tessdata", fetch=fetch)
    assert asked == ["fra"]
    assert path.read_bytes() == ENG_PACK


def test_add_language_rejects_unknown_codes_and_auto(tmp_path: Path) -> None:
    for code in ("xx", "auto"):
        with pytest.raises(LanguagePackError, match=code):
            add_language(code, tmp_path, fetch=lambda pack: ENG_PACK)
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


def test_add_language_rejects_a_file_that_is_not_a_tesseract_pack(tmp_path: Path) -> None:
    junk = tmp_path / "junk.traineddata"
    junk.write_bytes(b"not a pack")
    tessdata = tmp_path / "tessdata"
    with pytest.raises(LanguagePackError, match="not a valid Tesseract pack"):
        add_language("de", tessdata, from_file=junk)
    with pytest.raises(LanguagePackError, match="not a valid Tesseract pack"):
        add_language("de", tessdata, fetch=lambda pack: b"<html>404</html>")
    assert not tessdata.exists() or not list(tessdata.iterdir())


def test_a_failed_install_leaves_no_partial_file(tmp_path: Path) -> None:
    tessdata = tmp_path / "tessdata"
    (tessdata / "deu.traineddata").mkdir(parents=True)  # the final rename can't succeed
    with pytest.raises(OSError):
        add_language("de", tessdata, fetch=lambda pack: ENG_PACK)
    assert [p.name for p in tessdata.iterdir()] == ["deu.traineddata"]
