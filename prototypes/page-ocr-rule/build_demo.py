"""PROTOTYPE (throwaway) — inline signals.json into the single-file demo.

    python3 prototypes/page-ocr-rule/build_demo.py   # writes page-ocr-rule.html
"""
from pathlib import Path

HERE = Path(__file__).parent
html = (HERE / "demo.template.html").read_text()
data = (HERE / "signals.json").read_text().replace("</", "<\\/")
(HERE / "page-ocr-rule.html").write_text(html.replace("__SIGNALS__", data))
print("wrote", HERE / "page-ocr-rule.html")
