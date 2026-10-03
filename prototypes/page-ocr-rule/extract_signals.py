"""PROTOTYPE (throwaway) — measure per-Page signals for the "send this Page to OCR?" rule.

Reads a handful of corpus PDFs (born-digital and synthetic scans) plus the tricky
cases from make_tricky.py, and writes signals.json next to this file: one row per
Page with cheap text-layer / object signals, a thumbnail, and a Jev Noul on
whether the text layer reads as usable text.

    uv run --no-project --with pypdfium2 --with pillow --with typesafe-sdk \
        prototypes/page-ocr-rule/extract_signals.py
"""

import base64
import csv
import io
import json
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as raw

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
PDF = ROOT / "corpus" / "data" / "pdf"
MODEL = "jev-1.13.0"
CORPUS = [  # (stem, expect)
    ("aapl-20250927", "keep"),
    ("gplb_10k", "keep"),
    ("ko-20260703", "keep"),
    ("hd-20260201.scan", "ocr"),
    ("crona_10k.scan", "ocr"),
]
GRID = 64
WORDS = {w.strip().lower() for w in open("/usr/share/dict/words")}
JEV_SAMPLE_CHARS = 2000


def in_dict(w: str) -> bool:
    if w in WORDS:
        return True
    for suf in ("s", "es", "ed", "ing", "ly", "ies"):
        if w.endswith(suf) and (w[: -len(suf)] in WORDS or w[: -len(suf)] + ("y" if suf == "ies" else "") in WORDS):
            return True
    return False


def text_signals(text: str) -> dict:
    chars = [c for c in text if not c.isspace()]
    bad = sum(
        1 for c in chars
        if c == "�" or unicodedata.category(c) in ("Cc", "Co", "Cn", "Cs")
    )
    tokens = [t.lower() for t in re.findall(r"[A-Za-z]{2,}", text)]
    known = sum(1 for t in tokens if in_dict(t))
    return {
        "chars": len(chars),
        "bad_char_ratio": round(bad / len(chars), 3) if chars else 0.0,
        "words": len(tokens),
        "word_ratio": round(known / len(tokens), 3) if tokens else 0.0,
    }


def object_signals(page) -> dict:
    pw, ph = page.get_size()
    mask = [[False] * GRID for _ in range(GRID)]
    images = paths = text_objs = invisible = 0
    for obj in page.get_objects():
        if obj.type == raw.FPDF_PAGEOBJ_IMAGE:
            images += 1
            l, b, r, t = obj.get_bounds()
            x0, x1 = max(0, int(l / pw * GRID)), min(GRID, int(r / pw * GRID + 0.999))
            y0, y1 = max(0, int(b / ph * GRID)), min(GRID, int(t / ph * GRID + 0.999))
            for y in range(y0, y1):
                for x in range(x0, x1):
                    mask[y][x] = True
        elif obj.type == raw.FPDF_PAGEOBJ_PATH:
            paths += 1
        elif obj.type == raw.FPDF_PAGEOBJ_TEXT:
            text_objs += 1
            if raw.FPDFTextObj_GetTextRenderMode(obj.raw) == raw.FPDF_TEXTRENDERMODE_INVISIBLE:
                invisible += 1
    return {
        "image_count": images,
        "image_coverage": round(sum(map(sum, mask)) / GRID**2, 3),
        "path_count": paths,
        "invisible_text_ratio": round(invisible / text_objs, 3) if text_objs else 0.0,
    }


def thumb(page) -> str:
    img = page.render(scale=0.28).to_pil().convert("RGB")
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=55)
    return base64.b64encode(b.getvalue()).decode()


def load_key() -> None:
    env = ROOT / ".env"
    if "TYPESAFE_API_KEY" not in os.environ and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                os.environ["TYPESAFE_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")


def jev_usable(rows: list[dict]) -> None:
    from typesafe_sdk import Noul, TypeSafeClient

    load_key()
    # Phrasing picked from three tried on the tricky pages: the "real words" framing separated
    # garbled / bad-OCR text (0.2–0.35) from good text, including French (≥0.76), best.
    q = {"usable": Noul(instructions="Are most of the words in `page_text` correctly spelled real words in some human language?")}
    todo = [r for r in rows if r["chars"] > 0]
    tokens = 0
    with TypeSafeClient(timeout=60.0) as client:
        def ask(r):
            for attempt in range(4):
                try:
                    resp = client.system_one(state={"page_text": r.pop("_sample")}, questions=q, model=MODEL)
                    return r, resp.answers["usable"].model_dump()["noul"], resp.usage.input_tokens
                except Exception as e:  # prototype: retry 429s / blips, then give up on this page
                    if attempt == 3:
                        print("jev failed:", r["id"], e, file=sys.stderr)
                        return r, None, 0
                    time.sleep(1 + attempt)

        with ThreadPoolExecutor(8) as pool:
            for r, p, t in pool.map(ask, todo):
                r["jev_usable"] = p
                tokens += t
    print(f"jev: {len(todo)} pages, {tokens} input tokens (≈${tokens * 0.042 / 1e6:.4f})")


def main() -> None:
    rows = []
    sources = [(next(PDF.glob(f"*{stem}.pdf")), expect, None) for stem, expect in CORPUS]
    cases = json.loads((PDF / "tricky" / "cases.json").read_text())
    sources += [(PDF / c["file"], c["expect"], c) for c in cases]
    for path, expect, meta in sources:
        doc = pdfium.PdfDocument(path)
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_textpage().get_text_range()
            t0 = time.perf_counter()
            sig = {**text_signals(text), **object_signals(page)}
            rows.append({
                "id": f"{path.stem}#{i + 1}",
                "doc": path.stem,
                "page": i + 1,
                "group": "tricky" if meta else ("scan" if expect == "ocr" else "born-digital"),
                "expect": expect,
                "what": meta["what"] if meta else "",
                "why": meta["why"] if meta else "",
                **sig,
                "signal_ms": round((time.perf_counter() - t0) * 1000, 2),
                "jev_usable": None,
                "text_preview": text[:280],
                "thumb": thumb(page),
                "_sample": text[:JEV_SAMPLE_CHARS],
            })
    if "--no-jev" not in sys.argv:
        jev_usable(rows)
    for r in rows:
        r.pop("_sample", None)
    (HERE / "signals.json").write_text(json.dumps(rows, ensure_ascii=False))
    w = csv.DictWriter(sys.stdout, ["id", "expect", "chars", "word_ratio", "bad_char_ratio", "image_coverage",
                                    "path_count", "invisible_text_ratio", "jev_usable"], extrasaction="ignore")
    w.writeheader()
    for r in rows:
        if r["group"] != "born-digital" or r["word_ratio"] < 0.8 or r["chars"] < 200:
            w.writerow(r)
    print(f"{len(rows)} pages", file=sys.stderr)


if __name__ == "__main__":
    main()
