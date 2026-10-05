# SPDX-License-Identifier: Apache-2.0
"""OCR languages (spec §4.3): the ISO 639-1 table, installed Tesseract packs and `add_language`.

The one table mapping a Job's `ocr_languages` to Tesseract packs and RapidOCR model families.
`jobs:validate`, extraction and the CLI all read it.
"""

import os
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

# ISO 639-1 -> Tesseract pack. A language outside this table is unknown.
TESSERACT_CODES = {
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "nl": "nld",
    "pl": "pol",
    "sv": "swe",
    "da": "dan",
    "no": "nor",
    "fi": "fin",
    "cs": "ces",
    "tr": "tur",
    "ru": "rus",
    "uk": "ukr",
    "el": "ell",
    "ar": "ara",
    "he": "heb",
    "hi": "hin",
    "ja": "jpn",
    "ko": "kor",
    "zh": "chi_sim",
}
# ISO 639-1 -> RapidOCR model family. The `quality` Profile bakes in PP-OCR `latin` only (spec §2),
# so only languages that family reads appear here.
RAPIDOCR_CODES = {
    code: "latin"
    for code in ("en", "de", "fr", "es", "it", "pt", "nl", "pl", "sv", "da", "no", "fi", "cs", "tr")
}

TESSDATA_FAST_URL = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/{pack}.traineddata"
Fetcher = Callable[[str], bytes]


class LanguagePackError(Exception):
    """A language pack can't be installed."""


def tesseract_code(language: str) -> str:
    return TESSERACT_CODES[language]


def rapidocr_code(language: str) -> str | None:
    return RAPIDOCR_CODES.get(language)


def workspace_tessdata(workspace: Path) -> Path:
    return workspace / "models" / "tessdata"


def installed_languages(tessdata_dirs: Iterable[Path]) -> set[str]:
    """ISO codes whose Tesseract pack is in any of the directories."""
    dirs = list(tessdata_dirs)
    return {
        code
        for code, pack in TESSERACT_CODES.items()
        if any((d / f"{pack}.traineddata").is_file() for d in dirs)
    }


def _download(pack: str) -> bytes:
    with urllib.request.urlopen(TESSDATA_FAST_URL.format(pack=pack), timeout=60) as response:
        data: bytes = response.read()
    return data


def add_language(
    language: str, tessdata: Path, *, from_file: Path | None = None, fetch: Fetcher = _download
) -> Path:
    """Install a `tessdata_fast` pack into `tessdata`, from `from_file` or by download."""
    if language not in TESSERACT_CODES:
        raise LanguagePackError(f"unknown language '{language}'")
    pack = TESSERACT_CODES[language]
    if from_file is not None:
        try:
            data = from_file.read_bytes()
        except OSError as err:
            raise LanguagePackError(f"could not read {from_file}: {err}") from err
    elif os.environ.get("ACCELEREAD_OFFLINE") == "1":
        raise LanguagePackError(
            f"ACCELEREAD_OFFLINE=1 forbids downloading '{pack}'; use --from-file"
        )
    else:
        try:
            data = fetch(pack)
        except OSError as err:
            raise LanguagePackError(f"could not download '{pack}': {err}") from err
    if not data:
        raise LanguagePackError(f"the '{pack}' pack is empty")
    tessdata.mkdir(parents=True, exist_ok=True)
    target = tessdata / f"{pack}.traineddata"
    partial = target.with_suffix(".partial")
    partial.write_bytes(data)
    partial.replace(target)
    return target
