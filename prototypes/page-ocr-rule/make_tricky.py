"""PROTOTYPE (throwaway) — synthesise PDF Pages that stress the "send this Page to OCR?" rule.

The dogfood corpus PDFs are all-text or all-scan; real trouble lives in between.
Writes one PDF per case to corpus/data/pdf/tricky/ (gitignored) plus cases.json
with the expected outcome for each.

    uv run --no-project --with pypdfium2 --with pillow --with pikepdf --with reportlab \
        --with matplotlib prototypes/page-ocr-rule/make_tricky.py
"""

import glob
import io
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pikepdf
import pypdfium2 as pdfium
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from PIL import Image, ImageFilter
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "corpus" / "data" / "pdf"
OUT = PDF / "tricky"
W, H = letter
rng = random.Random(7)


def src(stem: str) -> Path:
    return Path(glob.glob(str(PDF / f"*{stem}.pdf"))[0])


def page_image(path: Path, i: int, scale=150 / 72, scanned=True) -> Image.Image:
    img = pdfium.PdfDocument(path)[i].render(scale=scale).to_pil().convert("L")
    if scanned:  # same flavour as corpus/make_pdfs.py: slight skew, blur, noise
        img = img.rotate(rng.uniform(-1.2, 1.2), fillcolor=255, expand=False).filter(ImageFilter.GaussianBlur(0.6))
        noise = Image.effect_noise(img.size, 18)
        img = Image.blend(img, noise, 0.08)
    return img


def page_text(path: Path, i: int) -> str:
    return pdfium.PdfDocument(path)[i].get_textpage().get_text_range()


def draw_lines(c, text, x=72, y=H - 72, size=10, leading=13, invisible=False, width_chars=95):
    t = c.beginText(x, y)
    t.setFont("Helvetica", size)
    t.setLeading(leading)
    if invisible:
        t.setTextRenderMode(3)
    for para in text.splitlines():
        while para:
            t.textLine(para[:width_chars])
            para = para[width_chars:]
    c.drawText(t)


def corrupt(text: str, rate: float) -> str:
    out = []
    for ch in text:
        if ch.isalnum() and rng.random() < rate:
            out.append(rng.choice("il1|!rnmvwcoe0@#%&;:~^"))
        else:
            out.append(ch)
    return "".join(out)


def save(c_or_bytes, name):
    path = OUT / f"{name}.pdf"
    if isinstance(c_or_bytes, bytes):
        path.write_bytes(c_or_bytes)
    else:
        c_or_bytes.save()
    return path.name


def new(name):
    return canvas.Canvas(str(OUT / f"{name}.pdf"), pagesize=letter)


def full_image(c, img):
    c.drawImage(ImageReader(img), 0, 0, W, H)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    aapl, ko, hd_scan, legend = src("aapl-20250927"), src("ko-20260703"), src("hd-20260201.scan"), src("legend_i10k-123125")
    cases = []

    def case(name, expect, what, why):
        cases.append({"file": f"tricky/{name}.pdf", "page": 0, "expect": expect, "what": what, "why": why})

    # 1. Broken font encoding: strip ToUnicode maps from a born-digital page.
    pdf = pikepdf.open(aapl)
    for i in range(len(pdf.pages) - 1, 0, -1):
        if i != 30:
            del pdf.pages[i]
    del pdf.pages[0]
    for obj in pdf.objects:
        if isinstance(obj, pikepdf.Dictionary) and obj.get("/Type") == "/Font" and "/ToUnicode" in obj:
            del obj["/ToUnicode"]
    buf = io.BytesIO()
    pdf.save(buf)
    save(buf.getvalue(), "garbled-encoding")
    case("garbled-encoding", "ocr", "Garbled text layer (font has no Unicode map)",
         "Looks fine on screen but the text layer is junk; only OCR recovers the words.")

    # 2. Scan with a typed Bates stamp / header on top.
    c = new("scan-with-stamp")
    full_image(c, page_image(hd_scan, 3, scanned=False))
    c.setFont("Helvetica-Bold", 8)
    c.drawString(40, H - 24, "CONFIDENTIAL — PRODUCED UNDER PROTECTIVE ORDER")
    c.drawString(W - 140, 20, "HD-0004417")
    c.save()
    case("scan-with-stamp", "ocr", "Scan with a typed stamp on top",
         "A handful of real characters sit over a full-page scan; the body is only in the image.")

    # 3/4. Scans that already carry an earlier OCR layer (invisible text), good vs bad.
    text = page_text(aapl, 20)
    for name, rate, expect, what, why in [
        ("scan-good-ocr-layer", 0.01, "keep", "Scan with a good earlier OCR layer",
         "Someone already OCR'd it well; re-OCRing costs time and gains nothing."),
        ("scan-bad-ocr-layer", 0.35, "ocr", "Scan with a poor earlier OCR layer",
         "The hidden OCR text is mostly wrong; Tesseract would do better."),
    ]:
        c = new(name)
        full_image(c, page_image(aapl, 20))
        draw_lines(c, corrupt(text, rate), invisible=True, size=9, leading=11.5, width_chars=110)
        c.save()
        case(name, expect, what, why)

    # 5. Text drawn as vector outlines (no text layer, no images).
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 612)
    ax.set_ylim(0, 792)
    ax.axis("off")
    body = page_text(ko, 11).splitlines()
    fp = FontProperties(family="DejaVu Sans")
    y = 740
    for line in [ln for ln in body if ln.strip()][:48]:
        tp = TextPath((60, y), line[:100], size=8.5, prop=fp)
        ax.add_patch(PathPatch(tp, color="black", lw=0))
        y -= 14
    fig.savefig(OUT / "text-as-outlines.pdf")
    plt.close(fig)
    case("text-as-outlines", "ocr", "Text drawn as vector shapes",
         "Designed reports often convert fonts to outlines: no characters and no images, only paths.")

    # 6. Glossy page: full-bleed background photo with real text on top.
    bg = Image.effect_mandelbrot((850, 1100), (-2.2, -1.4, 1.0, 1.4), 60).convert("RGB")
    bg = Image.blend(bg, Image.new("RGB", bg.size, (230, 236, 245)), 0.75)
    c = new("glossy-background")
    full_image(c, bg)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(72, H - 90, "Letter to Shareholders")
    draw_lines(c, page_text(aapl, 22), y=H - 130, size=10)
    c.save()
    case("glossy-background", "keep", "Full-bleed background image with real text",
         "Image covers the whole page, but the text layer is complete and correct.")

    # 7. Half text, half scanned table.
    c = new("half-text-half-scan")
    draw_lines(c, page_text(ko, 29)[:2200], size=9.5)
    tbl = page_image(ko, 7).crop((0, 300, 1275, 1100))
    c.drawImage(ImageReader(tbl), 36, 36, W - 72, (W - 72) * tbl.height / tbl.width)
    c.save()
    case("half-text-half-scan", "ocr", "Half typed text, half scanned exhibit",
         "The typed half is fine but the scanned table carries numbers the text layer lacks.")

    # 8. Chart page: big chart image, caption and a paragraph.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(["2021", "2022", "2023", "2024", "2025"], [365, 394, 383, 391, 416], color="#4a6fa5")
    ax.set_title("Net sales ($bn)")
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=110)
    plt.close(fig)
    c = new("chart-page")
    c.drawImage(ImageReader(io.BytesIO(b.getvalue())), 54, H - 470, W - 108, 400)
    draw_lines(c, "Figure 3. Net sales by fiscal year.\n\n" + page_text(aapl, 24)[:1600], y=H - 500, size=10)
    c.save()
    case("chart-page", "keep", "Chart image with caption and paragraph",
         "A big picture plus real text; the chart's numbers are nice-to-have, not body text.")

    # 9. A scan stored as horizontal tiles.
    c = new("scan-tiled")
    img = page_image(hd_scan, 6, scanned=False)
    n = 10
    th = img.height // n
    for k in range(n):
        tile = img.crop((0, k * th, img.width, (k + 1) * th))
        c.drawImage(ImageReader(tile), 0, H - (k + 1) * H / n, W, H / n)
    c.save()
    case("scan-tiled", "ocr", "Scan stored as ten image strips",
         "No single image covers the page; only the union does.")

    # 10–12. Short pages that are fine as they are.
    c = new("blank-intentional")
    c.setFont("Helvetica-Oblique", 11)
    c.drawCentredString(W / 2, H / 2, "This page intentionally left blank.")
    c.save()
    case("blank-intentional", "keep", "\"Intentionally left blank\" page",
         "Very few characters, but nothing to recover.")

    c = new("blank-empty")
    c.showPage()
    c.save()
    case("blank-empty", "keep", "Truly empty page", "No text, no images: OCR would find nothing.")

    logo = Image.effect_mandelbrot((300, 300), (-2, -1.5, 1, 1.5), 40).convert("RGB")
    c = new("cover-page")
    c.drawImage(ImageReader(logo), W / 2 - 75, H - 300, 150, 150)
    c.setFont("Helvetica-Bold", 28)
    c.drawCentredString(W / 2, H - 360, "Legend Spices, Inc.")
    c.setFont("Helvetica", 16)
    c.drawCentredString(W / 2, H - 390, "Annual Report 2025")
    c.save()
    case("cover-page", "keep", "Cover page with logo", "A few words and a small image; the text layer has it all.")

    c = new("signature-page")
    draw_lines(c, page_text(legend, len(pdfium.PdfDocument(legend)) - 1)[:700], size=10)
    c.drawImage(ImageReader(page_image(hd_scan, 0).crop((100, 100, 600, 220))), 72, 300, 200, 48)
    c.save()
    case("signature-page", "keep", "Signature page with a signature image",
         "Short text plus a small scanned signature.")

    # 13. Non-English born-digital page.
    fr = ("Rapport annuel 2025. Le chiffre d'affaires consolidé du groupe s'établit à 4,2 milliards d'euros, "
          "en hausse de 6,3 % à périmètre et taux de change constants. La marge opérationnelle courante progresse "
          "de 80 points de base, portée par la hausse des volumes et la maîtrise des coûts. Le conseil "
          "d'administration proposera à l'assemblée générale un dividende de 1,85 euro par action. ") * 6
    c = new("french-report")
    draw_lines(c, fr, size=10)
    c.save()
    case("french-report", "keep", "French annual report page",
         "Perfect text layer, but few words are in an English dictionary.")

    (OUT / "cases.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False))
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
