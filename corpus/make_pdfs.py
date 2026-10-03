"""Derive PDF test Documents from the EDGAR HTML corpus.

- born-digital: HTML printed to PDF with headless Chrome (has a text layer).
- scanned: those PDFs rasterised, skewed, noised and JPEG-compressed into
  image-only PDFs (no text layer), to exercise the OCR path.

Appends rows to manifest.csv. Run after fetch.py:
    uv run --with pypdfium2 --with pillow corpus/make_pdfs.py
"""

import csv
import random
import subprocess
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageFilter

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PDF_DIR = DATA / "pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BORN_DIGITAL = ["aapl-", "msft-", "nvda-", "nyt-", "gplb_", "legend_"]
SCANNED = ["hd-", "nke-", "ko-", "crona_"]
SCAN_MAX_PAGES = 15
random.seed(7)


def print_to_pdf(html: Path, pdf: Path) -> None:
    if not pdf.exists():
        subprocess.run(
            [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={pdf}", html.resolve().as_uri()],
            check=True, capture_output=True, timeout=300,
        )


def fake_scan(src: Path, dest: Path) -> int:
    doc = pdfium.PdfDocument(src)
    pages = []
    for i in range(min(len(doc), SCAN_MAX_PAGES)):
        img = doc[i].render(scale=150 / 72).to_pil().convert("L")
        img = img.rotate(random.uniform(-1.5, 1.5), expand=True, fillcolor=255)
        noise = Image.effect_noise(img.size, 18).convert("L")
        img = Image.blend(img, noise, 0.08).filter(ImageFilter.GaussianBlur(0.4))
        pages.append(img)
    pages[0].save(dest, save_all=True, append_images=pages[1:], resolution=150, quality=70)
    return len(pages)


def main() -> None:
    PDF_DIR.mkdir(exist_ok=True)
    with open(ROOT / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    html_rows = [r for r in rows if r["format"] == "htm"]
    rows = [r for r in rows if r["format"] != "pdf"]  # idempotent re-runs

    def pick(prefix: str) -> dict:
        return next(r for r in html_rows if r["file"].split("_", 2)[2].startswith(prefix))

    for prefix in BORN_DIGITAL + SCANNED:
        src = pick(prefix)
        stem = Path(src["file"]).stem
        digital = PDF_DIR / f"{stem}.pdf"
        print_to_pdf(DATA / src["file"], digital)
        if prefix in BORN_DIGITAL:
            path, note = digital, "synthetic born-digital: Chrome print of the HTML filing"
        else:
            path = PDF_DIR / f"{stem}.scan.pdf"
            n = fake_scan(digital, path)
            note = f"synthetic scan: first {n} pages rasterised at 150dpi, skewed and noised, no text layer"
        rows.append({**src, "file": f"pdf/{path.name}", "format": "pdf", "notes": note})
        print(f"{path.name}  {src['sector']}  {src['form']}")

    with open(ROOT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} documents in manifest")


if __name__ == "__main__":
    main()
