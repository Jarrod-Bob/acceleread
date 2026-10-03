"""Build the acceleread dogfood corpus from SEC EDGAR.

Downloads the primary document of recent filings for the companies in
companies.csv, plus a handful of going-concern 10-Ks found via EDGAR
full-text search, into corpus/data/ (gitignored), and writes
corpus/manifest.csv with ground-truth labels.

SEC requires a contact in the User-Agent (https://www.sec.gov/os/accessing-edgar-data):
    ACCELEREAD_SEC_CONTACT="Your Name you@example.com" python corpus/fetch.py

Stdlib only. Stays well under SEC's 10 requests/second limit.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
FORM_ROTATION = ["10-K", "10-Q", "8-K", "10-K"]
GOING_CONCERN_COUNT = 6
GOING_CONCERN_SINCE = "2025-01-01"
DOC_TYPE = {
    "10-K": "annual-report",
    "10-Q": "quarterly-report",
    "8-K": "current-report",
    "20-F": "annual-report",
    "DEF 14A": "proxy-statement",
}

_last_request = 0.0


def get(url: str) -> bytes:
    global _last_request
    wait = 0.15 - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        _last_request = time.monotonic()
        return resp.read()


def sector_from_sic(sic: int) -> str:
    """Map an SEC SIC code to acceleread's test sector Taxonomy (approximate)."""
    if sic == 3021:  # rubber & plastics footwear (e.g. Nike)
        return "consumer-discretionary"
    if 1311 <= sic <= 1389 or 2900 <= sic <= 2999:
        return "energy"
    if 1000 <= sic <= 1499 or 2400 <= sic <= 2509 or 2600 <= sic <= 2699:
        return "materials"
    if 2800 <= sic <= 2829 or 2850 <= sic <= 2899 or 3000 <= sic <= 3399:
        return "materials" if sic != 3140 else "consumer-discretionary"
    if 2830 <= sic <= 2836 or 3840 <= sic <= 3851 or 8000 <= sic <= 8099:
        return "health-care"
    if 100 <= sic <= 999 or 2000 <= sic <= 2199 or 2840 <= sic <= 2844:
        return "consumer-staples"
    if 5400 <= sic <= 5499 or sic == 5912:
        return "consumer-staples"
    if 3570 <= sic <= 3579 or (3600 <= sic <= 3699 and sic != 3630):
        return "technology"
    if 3800 <= sic <= 3899 or 7370 <= sic <= 7379:
        return "technology"
    if 2700 <= sic <= 2799 or 4800 <= sic <= 4899 or 7800 <= sic <= 7899:
        return "communication"
    if 4900 <= sic <= 4999:
        return "utilities"
    if 6500 <= sic <= 6553 or sic == 6798:
        return "real-estate"
    if 6000 <= sic <= 6799:
        return "financials"
    if sic in (3711, 3713, 3714, 3716, 3751, 3630) or 2200 <= sic <= 2399:
        return "consumer-discretionary"
    if 2510 <= sic <= 2599 or 3900 <= sic <= 3999 or 7000 <= sic <= 7099:
        return "consumer-discretionary"
    if 5200 <= sic <= 5999:
        return "consumer-discretionary"
    if 1500 <= sic <= 1799 or 3400 <= sic <= 3799 or 4000 <= sic <= 4799:
        return "industrials"
    if 5000 <= sic <= 5199 or 7200 <= sic <= 8999:
        return "industrials"
    return "other"


def submissions(cik: int) -> dict:
    return json.loads(get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))


def latest_filing(subs: dict, form: str) -> dict | None:
    recent = subs["filings"]["recent"]
    for i, f in enumerate(recent["form"]):
        if f == form and recent["primaryDocument"][i]:
            return {
                "form": f,
                "accession": recent["accessionNumber"][i],
                "filed": recent["filingDate"][i],
                "primary_doc": recent["primaryDocument"][i],
            }
    return None


def download(cik: int, filing: dict) -> Path:
    acc = filing["accession"].replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{filing['primary_doc']}"
    dest = DATA / f"{cik}_{acc}_{filing['primary_doc']}"
    if not dest.exists():
        dest.write_bytes(get(url))
    filing["url"] = url
    return dest


def going_concern_filings() -> list[tuple[int, str, dict]]:
    """Recent 10-K primary documents containing going-concern doubt language."""
    q = urllib.parse.quote('"substantial doubt" "going concern"')
    url = (
        f"https://efts.sec.gov/LATEST/search-index?q={q}&forms=10-K"
        f"&dateRange=custom&startdt={GOING_CONCERN_SINCE}&enddt=2099-12-31"
    )
    picked: dict[int, tuple[int, str, dict]] = {}
    for h in json.loads(get(url))["hits"]["hits"]:
        src = h["_source"]
        if src.get("file_type") != "10-K":
            continue  # skip exhibits that merely mention the phrase
        cik = int(src["ciks"][0])
        filing = {
            "form": "10-K",
            "accession": src["adsh"],
            "filed": src["file_date"],
            "primary_doc": h["_id"].split(":", 1)[1],
        }
        picked.setdefault(cik, (cik, src["display_names"][0], filing))
        if len(picked) == GOING_CONCERN_COUNT:
            break
    return list(picked.values())


def row(cik: int, name: str, subs: dict, filing: dict, path: Path, expected: str, notes: str) -> dict:
    sic = int(subs.get("sic") or 0)
    sector = sector_from_sic(sic)
    if expected and expected != sector:
        notes = f"{notes}; SIC-derived sector differs from expected {expected}".lstrip("; ")
    return {
        "file": path.name,
        "company": name,
        "cik": cik,
        "sic": sic,
        "sector": sector,
        "doc_type": DOC_TYPE.get(filing["form"], "other"),
        "form": filing["form"],
        "filed": filing["filed"],
        "format": path.suffix.lstrip(".").lower(),
        "going_concern": "",
        "url": filing["url"],
        "notes": notes,
    }


def main() -> None:
    DATA.mkdir(exist_ok=True)
    tickers = {
        v["ticker"]: (int(v["cik_str"]), v["title"])
        for v in json.loads(get("https://www.sec.gov/files/company_tickers.json")).values()
    }
    rows = []
    with open(ROOT / "companies.csv") as f:
        for i, c in enumerate(csv.DictReader(f)):
            cik, name = tickers[c["ticker"]]
            if c.get("cik"):  # ticker list can point at a new entity with no filings yet
                cik = int(c["cik"])
            subs = submissions(cik)
            filing = latest_filing(subs, FORM_ROTATION[i % len(FORM_ROTATION)])
            if not filing:
                print(f"skip {c['ticker']}: no filing", file=sys.stderr)
                continue
            rows.append(row(cik, name, subs, filing, download(cik, filing), c["expected_sector"], ""))
            print(f"{c['ticker']:6} {filing['form']:5} {rows[-1]['sector']}")
    for cik, name, filing in going_concern_filings():
        subs = submissions(cik)
        r = row(cik, name, subs, filing, download(cik, filing), "", "found via going-concern full-text search")
        r["going_concern"] = "likely"
        rows.append(r)
        print(f"GC     10-K  {r['sector']}  {name}")
    with open(ROOT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} documents -> {ROOT / 'manifest.csv'}")


if __name__ == "__main__":
    contact = os.environ.get("ACCELEREAD_SEC_CONTACT")
    if not contact:
        sys.exit("Set ACCELEREAD_SEC_CONTACT='Your Name you@example.com' (SEC fair-access policy).")
    USER_AGENT = f"acceleread-corpus {contact}"
    main()
