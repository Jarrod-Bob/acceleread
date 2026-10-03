"""Live Jev probe on a few corpus Documents (acceleread wayfinder task).

Answers three facts later tickets depend on:
  1. What Jev returns when state + question exceeds its 32k-token budget.
  2. Observed characters per token on filing text (Jev has no tokenizer).
  3. Whether one request with sector + doc-type Choices and finance Questions
     gives sane answers on a handful of filings.

Reads TYPESAFE_API_KEY from the environment or ~/acceleread/.env. Writes
corpus/probe-results.json (no secrets). Costs well under one cent.
    uv run --no-project --with typesafe-sdk corpus/probe_jev.py
"""

import csv
import json
import os
import re
from html.parser import HTMLParser
from pathlib import Path

from typesafe_sdk import Choice, Noul, Score, TypeSafeAPIError, TypeSafeClient

ROOT = Path(__file__).parent
MODEL = "jev-1.13.0"
MAX_STATE_CHARS = 80_000  # ~23k tokens at a pessimistic 3.5 chars/token
OVER_LIMIT_CHARS = 250_000
SAMPLE = ["aapl-", "jpm-", "nvda-", "pld-", "crona_", "gplb_"]

SECTORS = {
    "technology": "Software, semiconductors, computer hardware, IT services",
    "communication": "Telecom carriers, media, publishing, entertainment, internet platforms",
    "consumer-discretionary": "Retail, autos, apparel, hotels, restaurants, leisure, household durables",
    "consumer-staples": "Food, beverages, tobacco, household and personal products, grocery and drug retail",
    "energy": "Oil and gas exploration, production, refining, oilfield services",
    "financials": "Banks, insurers, asset managers, brokers, payment and lending companies",
    "health-care": "Pharmaceuticals, biotech, medical devices, hospitals and health services",
    "industrials": "Machinery, aerospace and defence, transport and logistics, construction, business services",
    "materials": "Chemicals, metals and mining, paper and packaging, construction materials",
    "real-estate": "REITs and real-estate owners, developers and services",
    "utilities": "Electric, gas and water utilities",
    "other": "A shell company, blank-check company, or a business fitting none of the sectors above",
}
DOC_TYPES = {
    "annual-report": "An annual report on the company's full fiscal year, e.g. Form 10-K or 20-F",
    "quarterly-report": "An interim report on a fiscal quarter, e.g. Form 10-Q",
    "current-report": "A short report announcing a specific event, e.g. Form 8-K with earnings or a material agreement",
    "proxy-statement": "A notice of shareholder meeting and matters to vote on",
    "other": "Any other kind of document",
}


def questions() -> dict:
    return {
        "sector": Choice(
            instructions="Which sector best describes the main business of the company that filed `filing`?",
            criteria=SECTORS,
        ),
        "doc_type": Choice(
            instructions="What kind of document is `filing`?",
            criteria=DOC_TYPES,
        ),
        "going_concern": Noul(
            instructions="Does `filing` state that there is substantial doubt about the company's ability to continue as a going concern?",
        ),
        "material_weakness": Noul(
            instructions="Does `filing` disclose a material weakness in the company's internal control over financial reporting?",
        ),
        "outlook": Score(
            instructions="How does `filing` characterise the company's near-term business outlook?",
            criteria=[
                "Clearly negative: losses, shrinking business, financing trouble or doubt about survival",
                "Cautious: notable headwinds or risks dominate the discussion",
                "Mixed or neutral: no clear lean either way",
                "Positive: growth or improving results are emphasised",
                "Strongly positive: strong growth and confident forward statements",
            ],
        ),
    }


class _Text(HTMLParser):
    """Visible text of an EDGAR HTML filing, skipping the hidden inline-XBRL header."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("ix:header", "script", "style") or ("style", "display:none") in attrs:
            self._skip += 1

    def handle_endtag(self, tag):
        if self._skip and tag in ("ix:header", "script", "style", "div"):
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_text(path: Path) -> str:
    p = _Text()
    p.feed(path.read_text(errors="ignore"))
    return re.sub(r"\s+", " ", " ".join(p.parts)).strip()


def load_key() -> None:
    env = ROOT.parent / ".env"
    if "TYPESAFE_API_KEY" not in os.environ and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                os.environ["TYPESAFE_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")


def main() -> None:
    load_key()
    rows = {r["file"]: r for r in csv.DictReader(open(ROOT / "manifest.csv")) if r["format"] == "htm"}
    results: dict = {"model": MODEL, "documents": []}
    with TypeSafeClient(timeout=120.0) as client:
        for prefix in SAMPLE:
            row = next(r for f, r in rows.items() if f.split("_", 2)[2].startswith(prefix))
            text = html_text(ROOT / "data" / row["file"])
            sent = text[:MAX_STATE_CHARS]
            resp = client.system_one(
                state={"filing": {"company": row["company"], "text": sent}},
                questions=questions(),
                model=MODEL,
            )
            a = {k: v.model_dump() for k, v in resp.answers.items()}
            chars = len(json.dumps(sent)) + len(row["company"])
            results["documents"].append({
                "file": row["file"], "label_sector": row["sector"], "label_doc_type": row["doc_type"],
                "label_going_concern": row["going_concern"], "full_chars": len(text), "sent_chars": len(sent),
                "input_tokens": resp.usage.input_tokens, "chars_per_token": round(chars / resp.usage.input_tokens, 2),
                "model": resp.model, "answers": a,
            })
            print(f"{prefix:8} tokens={resp.usage.input_tokens:6} sector={a['sector'].get('choice')} "
                  f"(label {row['sector']}) doc={a['doc_type'].get('choice')} "
                  f"gc={a['going_concern'].get('noul')} mw={a['material_weakness'].get('noul')} "
                  f"outlook={a['outlook'].get('score')}")

        big = html_text(ROOT / "data" / next(f for f in rows if "_jpm-" in f))[:OVER_LIMIT_CHARS]
        try:
            resp = client.system_one(state={"filing": {"text": big}}, questions={"sector": questions()["sector"]}, model=MODEL)
            results["over_limit"] = {"chars": len(big), "outcome": "accepted", "input_tokens": resp.usage.input_tokens}
        except TypeSafeAPIError as e:
            results["over_limit"] = {"chars": len(big), "outcome": "error", "status": e.status, "message": str(e)[:500]}
        print("over-limit:", results["over_limit"])
    (ROOT / "probe-results.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
