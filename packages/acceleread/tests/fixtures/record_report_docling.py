# SPDX-License-Identifier: Apache-2.0
"""Re-record report.docling.json, Docling's output for report.pdf (needs `models fetch`).

uv run --extra quality python \
    packages/acceleread/tests/fixtures/record_report_docling.py [WORKSPACE]
"""

import sys
from pathlib import Path

from acceleread import quality
from acceleread.quality_models import models_dir
from acceleread.workspace import resolve_workspace_path

HERE = Path(__file__).parent


def main() -> None:
    workspace = Path(sys.argv[1]) if len(sys.argv) > 1 else resolve_workspace_path(None)
    converter = quality._converter(models_dir(workspace), ("en",), ocr=False, tables=False)
    result = converter.convert(HERE / "report.pdf", page_range=(1, 3))
    out = HERE / "report.docling.json"
    result.document.save_as_json(out)
    print(out)


if __name__ == "__main__":
    main()
