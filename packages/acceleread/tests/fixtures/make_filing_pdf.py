# SPDX-License-Identifier: Apache-2.0
"""Regenerate filing10k.pdf, a three-page 10-K-shaped PDF with Items 1, 1A and 7.

uv run --no-project --with reportlab python packages/acceleread/tests/fixtures/make_filing_pdf.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas

FILLER = "The company continues to operate across its segments and markets worldwide."
PAGES = [
    [
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
        "FORM 10-K",
        "Item 1. Business",
        "Northwind Solar designs and manufactures photovoltaic panels.",
        FILLER,
        FILLER,
    ],
    [
        "Item 1A. Risk Factors",
        "Polysilicon supply shortages could halt production of our panels.",
        FILLER,
        FILLER,
    ],
    [
        "Item 7. Management's Discussion and Analysis",
        "Revenue grew 18 percent on higher panel shipments.",
        FILLER,
        FILLER,
    ],
]


def main() -> None:
    out = Path(__file__).with_name("filing10k.pdf")
    canvas = Canvas(str(out), pagesize=letter, invariant=True)
    canvas.setTitle("Northwind Solar 10-K")
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
