# SPDX-License-Identifier: Apache-2.0
"""`acceleread doctor`: Tesseract, language packs, models and extras (spec §2).

It only reads: it never creates the Workspace or downloads anything. Optional parts that aren't
installed are reported as `absent`, which isn't a failure. Only a broken core (no Tesseract, no
vendored English pack) fails.
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from acceleread.extract import VENDORED_TESSDATA, tesseract_version
from acceleread.languages import installed_languages, offline, workspace_tessdata

Status = Literal["ok", "absent", "error"]
# PROVISIONAL (#40, the `quality` Profile, owns the real layout): the Workspace `models/`
# subdirectories and import names probed for Docling, PP-OCR and the `[quality]` extra.
MODEL_DIRS = {"Docling weights": "docling", "PP-OCR weights": "ppocr"}
EXTRAS = {"quality": ("docling", "rapidocr"), "llm": ("anthropic",), "edgar": ("edgar",)}


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str


def _tesseract() -> Check:
    try:
        return Check("tesseract", "ok", tesseract_version())
    except Exception as err:
        return Check("tesseract", "error", f"unavailable: {err}")


def _packs(workspace: Path, vendored_dir: Path) -> list[Check]:
    vendored = installed_languages([vendored_dir])
    extra = installed_languages([workspace_tessdata(workspace)]) - vendored
    if "en" in vendored:
        core = Check("language packs (vendored)", "ok", ", ".join(sorted(vendored)))
    else:
        core = Check("language packs (vendored)", "error", "eng.traineddata is missing; reinstall")
    detail = ", ".join(sorted(extra)) if extra else "none; `acceleread ocr add-language xx`"
    return [core, Check("language packs (workspace)", "ok" if extra else "absent", detail)]


def _models(workspace: Path) -> list[Check]:
    checks = []
    for label, directory in MODEL_DIRS.items():
        found = (workspace / "models" / directory).is_dir()
        detail = "installed" if found else "absent (optional, `quality` Profile)"
        checks.append(Check(f"models: {label}", "ok" if found else "absent", detail))
    return checks


def _extras() -> list[Check]:
    checks = []
    for extra, modules in EXTRAS.items():
        missing = [m for m in modules if importlib.util.find_spec(m) is None]
        if missing:
            checks.append(Check(f"extra [{extra}]", "absent", f"absent: {', '.join(missing)}"))
        else:
            checks.append(Check(f"extra [{extra}]", "ok", "installed"))
    return checks


def run_checks(workspace: Path, vendored: Path = VENDORED_TESSDATA) -> list[Check]:
    return [
        Check("workspace", "ok" if workspace.is_dir() else "absent", str(workspace)),
        _tesseract(),
        *_packs(workspace, vendored),
        *_models(workspace),
        *_extras(),
        Check("offline mode", "ok", "on (ACCELEREAD_OFFLINE=1)" if offline() else "off"),
    ]


def report(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    return "\n".join(f"{c.status:<6} {c.name:<{width}}  {c.detail}" for c in checks)
