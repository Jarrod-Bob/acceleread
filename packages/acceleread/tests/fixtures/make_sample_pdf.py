# SPDX-License-Identifier: Apache-2.0
"""Regenerate sample.pdf, a two-page born-digital PDF about a fictional company.

uv run --no-project --with reportlab python packages/acceleread/tests/fixtures/make_sample_pdf.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas

PAGES = [
    [
        "NORTHWIND SOLAR, INC. - ANNUAL REPORT",
        "Item 1. Business",
        "Northwind Solar designs and manufactures photovoltaic panels and inverters",
        "for residential and commercial rooftops. We sell through installers in",
        "twelve states and operate two factories in Arizona.",
    ],
    [
        "Item 7. Management's Discussion and Analysis",
        "Revenue grew 18% to $412 million on higher panel shipments.",
        "Gross margin improved as polysilicon prices declined.",
        "We expect continued growth from utility-scale contracts next year.",
    ],
]


def main() -> None:
    out = Path(__file__).with_name("sample.pdf")
    canvas = Canvas(str(out), pagesize=letter, invariant=True)
    canvas.setTitle("Northwind Solar Annual Report")
    for lines in PAGES:
        y = 720
        for line in lines:
            canvas.drawString(72, y, line)
            y -= 18
        canvas.showPage()
    canvas.save()
    print(out)


if __name__ == "__main__":
    main()
