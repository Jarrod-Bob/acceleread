# SPDX-License-Identifier: Apache-2.0
"""Regenerate the OCR fixtures: scanned.pdf, mixed.pdf and text_beside_image.pdf.

uv run --no-project --with reportlab --with pillow python \
    packages/acceleread/tests/fixtures/make_ocr_pdfs.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

HERE = Path(__file__).parent
SCAN_LINES = [
    "Gross margin improved as polysilicon",
    "prices declined during the year.",
    "Revenue grew eighteen percent.",
]
TEXT_LINES = [
    "Northwind Solar designs and manufactures photovoltaic panels and inverters",
    "for residential and commercial rooftops. We sell through installers in",
    "twelve states and operate two factories in Arizona.",
]


def scan_image(lines: list[str]) -> Image.Image:
    """A page-sized image of printed text, as a scanner would produce."""
    image = Image.new("L", (1700, 2200), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=64)
    y = 200
    for line in lines:
        draw.text((120, y), line, fill=0, font=font)
        y += 120
    return image


def draw_scan(canvas: Canvas, lines: list[str]) -> None:
    width, height = letter
    canvas.drawImage(ImageReader(scan_image(lines)), 0, 0, width=width, height=height)
    canvas.showPage()


def draw_text(canvas: Canvas, lines: list[str]) -> None:
    y = 720
    for line in lines:
        canvas.drawString(72, y, line)
        y -= 18
    canvas.showPage()


def write(name: str) -> Canvas:
    return Canvas(str(HERE / name), pagesize=letter, invariant=True)


def main() -> None:
    canvas = write("scanned.pdf")
    draw_scan(canvas, SCAN_LINES)
    canvas.save()

    canvas = write("mixed.pdf")
    draw_text(canvas, TEXT_LINES)
    draw_scan(canvas, SCAN_LINES)
    canvas.save()

    canvas = write("text_beside_image.pdf")
    width, _ = letter
    canvas.drawImage(ImageReader(Image.new("L", (400, 400), 128)), 0, 0, width=width, height=600)
    y = 720
    for line in TEXT_LINES:
        canvas.drawString(72, y, line)
        y -= 18
    canvas.showPage()
    canvas.save()

    canvas = write("vector.pdf")
    for i in range(30):
        canvas.line(72, 100 + i * 20, 540, 100 + i * 20)
    canvas.showPage()
    canvas.save()


if __name__ == "__main__":
    main()
