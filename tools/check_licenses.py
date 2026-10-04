# SPDX-License-Identifier: Apache-2.0
"""Fail if any installed distribution has a licence outside the ADR 0003 allowlist.

Run inside the environment to check, e.g. after ``uv sync --no-dev --extra quality``:

    uv run --no-sync python tools/check_licenses.py

Licences come from PEP 639 ``License-Expression`` metadata, then ``License ::`` classifiers,
then the free-text ``License`` field. Packages that can't be classified, or are reviewed
exceptions, are listed in ``tools/license-exceptions.toml``.
"""

import re
import sys
import tomllib
from importlib.metadata import Distribution, distributions
from pathlib import Path

ALLOWED = {
    "0BSD",
    "BSL-1.0",
    "CC0-1.0",
    "CNRI-Python",
    "HPND",
    "MIT-CMU",
    "Zlib",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CDLA-Permissive-1.0",
    "CDLA-Permissive-2.0",
    "ISC",
    "MIT",
    "MIT-0",
    "MPL-2.0",
    "PSF-2.0",
    "Python-2.0",
}

CLASSIFIERS = {
    "Apache Software License": "Apache-2.0",
    "BSD License": "BSD-3-Clause",
    "ISC License (ISCL)": "ISC",
    "MIT License": "MIT",
    "MIT No Attribution License (MIT-0)": "MIT-0",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "Python Software Foundation License": "PSF-2.0",
    "Zero-Clause BSD (0BSD)": "0BSD",
}

# Free-text ``License`` fields, matched case-insensitively from the start of the field.
TEXT = [
    (r"apache(\s+license)?,?\s*(version\s*)?2(\.0)?|apache-2\.0|apache 2", "Apache-2.0"),
    (r"mit\b|the mit license", "MIT"),
    (r"bsd[- ]?3|3-clause bsd|new bsd|modified bsd", "BSD-3-Clause"),
    (r"bsd[- ]?2|2-clause bsd|simplified bsd", "BSD-2-Clause"),
    (r"bsd\b", "BSD-3-Clause"),
    (r"isc\b", "ISC"),
    (r"mpl[- ]?2(\.0)?|mozilla public license,? (version )?2\.0", "MPL-2.0"),
    (r"psf|python software foundation", "PSF-2.0"),
]

WORKSPACE = {"acceleread", "acceleread-eval"}
EXCEPTIONS_FILE = Path(__file__).with_name("license-exceptions.toml")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def expression_allowed(expr: str) -> bool:
    """Evaluate an SPDX expression: OR needs one allowed side, AND needs all."""
    expr = expr.strip().strip("()")
    if " OR " in expr:
        return any(expression_allowed(part) for part in expr.split(" OR "))
    if " AND " in expr:
        return all(expression_allowed(part) for part in expr.split(" AND "))
    return expr.split(" WITH ")[0].strip() in ALLOWED


def licences(dist: Distribution) -> list[str]:
    meta = dist.metadata
    if expr := meta.get("License-Expression"):
        return [expr]
    found = [
        CLASSIFIERS[c.split(" :: ")[-1]]
        for c in meta.get_all("Classifier") or []
        if c.startswith("License ::") and c.split(" :: ")[-1] in CLASSIFIERS
    ]
    if found:
        return found
    text = (meta.get("License") or "").strip().lower()
    for pattern, spdx in TEXT:
        if text and re.match(pattern, text):
            return [spdx]
    return []


def main() -> int:
    exceptions = {_norm(k): v for k, v in tomllib.loads(EXCEPTIONS_FILE.read_text()).items()}
    failures = []
    for dist in sorted(distributions(), key=lambda d: _norm(d.metadata["Name"])):
        name = _norm(dist.metadata["Name"])
        if name in WORKSPACE or name in exceptions:
            continue
        found = licences(dist)
        if not found:
            raw = [
                c.split(" :: ")[-1]
                for c in dist.metadata.get_all("Classifier") or []
                if c.startswith("License ::")
            ]
            hint = " / ".join(raw) or (dist.metadata.get("License") or "").strip()[:60]
            failures.append(f"{name} {dist.version}: not identified ({hint or 'no metadata'})")
        elif not any(expression_allowed(lic) for lic in found):
            failures.append(f"{name} {dist.version}: {' / '.join(found)}")
    if failures:
        print("Licences outside the ADR 0003 allowlist (add a reviewed entry to", end=" ")
        print(f"{EXCEPTIONS_FILE.name} only if acceptable):")
        print("\n".join(f"  {f}" for f in failures))
        return 1
    print(f"All {sum(1 for _ in distributions())} distributions within the allowlist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
