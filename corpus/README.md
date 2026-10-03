# Dogfood corpus

60 labelled Documents for evaluating acceleread: 50 SEC EDGAR filings (HTML) and 10 PDFs derived from them (6 born-digital, 4 synthetic scans with no text layer).

| File | What |
|---|---|
| `companies.csv` | 44 S&P 500 companies, 4 per sector, with expected sector (and a CIK override where the ticker map points at a new entity) |
| `manifest.csv` | One row per Document: sector (from SIC), doc_type, form, filed date, format, going-concern flag, source URL |
| `fetch.py` | Downloads the latest 10-K / 10-Q / 8-K per company in rotation, plus 6 recent 10-Ks containing going-concern doubt language |
| `make_pdfs.py` | Prints HTML filings to PDF (Chrome) and fakes scans from them |

Filings land in `data/` (gitignored). Rebuild:

```sh
ACCELEREAD_SEC_CONTACT="Your Name you@example.com" python3 corpus/fetch.py
uv run --no-project --with pypdfium2 --with pillow corpus/make_pdfs.py
```

**Label caveats:** sectors are derived from SEC SIC codes, which diverge from GICS-style sectors for some companies. `going_concern=likely` means the filing matched full-text search for going-concern language, not a hand-verified label. Real-world PDFs (glossy annual reports, non-US filings, genuine scans) are still to be added under `data/pdf/`.
