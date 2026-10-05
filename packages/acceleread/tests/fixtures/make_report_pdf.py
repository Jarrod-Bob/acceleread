# SPDX-License-Identifier: Apache-2.0
"""Regenerate report.pdf, a three-page born-digital report with real headings.

Docling's layout model needs styled headings and body paragraphs to emit `section_header` items.

uv run --no-project --with reportlab python packages/acceleread/tests/fixtures/make_report_pdf.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

BODY = (
    "Northwind Solar designs and manufactures photovoltaic panels and inverters for residential "
    "and commercial rooftops. We sell through independent installers in twelve states and "
    "operate two factories in Arizona. Our largest customers are regional installation "
    "companies, and no single customer accounted for more than ten percent of revenue. "
)
SECTIONS = [
    ("Item 1. Business", 1),
    ("Item 1A. Risk Factors", 2),
    ("Item 7. Management's Discussion and Analysis", 3),
]


def main() -> None:
    out = Path(__file__).with_name("report.pdf")
    styles = getSampleStyleSheet()
    story: list[object] = []
    for title, page in SECTIONS:
        story.append(Paragraph(title, styles["Heading1"]))
        story.append(Spacer(1, 12))
        for _ in range(3):
            story.append(Paragraph(BODY * 2, styles["BodyText"]))
            story.append(Spacer(1, 8))
        if page < len(SECTIONS):
            story.append(PageBreak())
    SimpleDocTemplate(str(out), pagesize=letter, invariant=True, title="Northwind report").build(
        story
    )
    print(out)


if __name__ == "__main__":
    main()
