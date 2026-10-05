# SPDX-License-Identifier: Apache-2.0
"""OCR languages (spec §4.3): the ISO 639-1 table, installed Tesseract packs and `add_language`.

The one table mapping a Job's `ocr_languages` to Tesseract packs and RapidOCR model families.
`jobs:validate`, extraction and the CLI all read it.
"""

import os
import tempfile
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


def offline() -> bool:
    """`ACCELEREAD_OFFLINE=1`: any runtime download is an error (spec §2)."""
    return os.environ.get("ACCELEREAD_OFFLINE") == "1"


def pack_file(tessdata: Path, language: str) -> Path:
    """Where a language's Tesseract pack lives in `tessdata` (an ISO 639-1 code)."""
    return tessdata / f"{TESSERACT_CODES[language]}.traineddata"


def workspace_tessdata(workspace: Path) -> Path:
    return workspace / "models" / "tessdata"


def installed_languages(tessdata_dirs: Iterable[Path]) -> set[str]:
    """ISO codes whose Tesseract pack is in any of the directories."""
    dirs = list(tessdata_dirs)
    return {code for code in TESSERACT_CODES if any(pack_file(d, code).is_file() for d in dirs)}


def _download(pack: str) -> bytes:
    with urllib.request.urlopen(TESSDATA_FAST_URL.format(pack=pack), timeout=60) as response:
        data: bytes = response.read()
    return data


def _loads(language: str, data: bytes) -> bool:
    """Whether Tesseract can load `data` as the language's pack."""
    import tesserocr

    with tempfile.TemporaryDirectory(prefix="acceleread-pack-") as scratch:
        pack_file(Path(scratch), language).write_bytes(data)
        try:
            tesserocr.PyTessBaseAPI(path=scratch, lang=TESSERACT_CODES[language]).End()
        except RuntimeError:
            return False
    return True


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
    elif offline():
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
    if not _loads(language, data):
        raise LanguagePackError(f"the '{pack}' data is not a valid Tesseract pack")
    tessdata.mkdir(parents=True, exist_ok=True)
    target = pack_file(tessdata, language)
    partial = target.with_suffix(".partial")
    try:
        partial.write_bytes(data)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target
