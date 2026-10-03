# How reliably can filing Sections be detected in EDGAR HTML and PDF reports?

Research for [#14](https://github.com/Jarrod-Bob/acceleread/issues/14), part of the v0 map ([#1](https://github.com/Jarrod-Bob/acceleread/issues/1)). Checked 2026-10-03/04 against SEC form instructions, the Docling and edgartools source code, GitHub/PyPI metadata, and an experiment on 20 real EDGAR filings. Token counts use `chars / 3.5` because Jev has no tokenizer (see [long-documents research](https://github.com/Jarrod-Bob/acceleread/blob/research/long-documents/research/long-documents.md)).

## Answer (TL;DR)

- **EDGAR HTML 10-K/10-Q: you can detect Sections reliably for most filers, but not with Docling or a plain regex.** The 20 filings in this experiment have **no `<h1>`–`<h6>` tags** in 18 cases. Docling's HTML backend only treats `h1`–`h6` as headings, so it returns **zero section headers** for typical filings. A naive "Item N." line regex got the correct key Sections in **21/30** cases. edgartools 5.60 (MIT) uses a TOC-, heading- and pattern-based hybrid detector and returned correct Items 1/1A/7 for about **31 of 39** key 10-K Sections and all 5 10-Q MD&As. The rest are hard by construction (see §3).
- **The hard cases are structural, not parsing bugs.** The SEC lets filers **incorporate items by reference**: Items 1–9A from the annual report to shareholders, filed as Exhibit 13, and Part III from the proxy. Filers may also replace the item layout with a **cross-reference index**. Wells Fargo's 10-K body is 300-character pointers. GE and Citi use cross-reference indexes. In these filings the "Section" is not in the primary document, or is not one contiguous span.
- **PDFs:** Docling's PDF layout model labelled **all 23 Item headings** as `section_header` on two 10-Ks printed to PDF, at 0.2–0.5 s/page on an M5 with OCR and tables off. A line regex over plain pypdfium2 text mis-split one of the two filings. Real glossy annual reports (non-US, no Item numbers) were **not tested**.
- **Inline XBRL does not mark item boundaries.** Its only useful signals are `dei:DocumentType` (form type), `…TextBlock` facts (financial-statement notes) and, since FY2024, `cyd:` tags covering Item 1C Cybersecurity.
- **Token sizes:** the Section that sector classification needs, **Item 1 Business, fits the ~30k state budget in about 89% of 10-Ks** (median about 11k tokens). Risk Factors fits about 73% of the time (median about 20k), MD&A about 93% (median about 13k), Item 8 only about 56%, and the **whole 10-K only about 17%** (median about 78k). Item 9A, where material weaknesses are disclosed, is tiny (median about 1.2k tokens).
- **Recommendation for v0:** a pluggable `SectionDetector` seam. Use edgartools' offline HTML parser behind an optional `acceleread[edgar]` extra for 10-K/10-Q/20-F HTML. For PDFs, use Docling `section_header` items plus an Item regex. Record a generic, confidence-scored **Section** on every Document Record. If detection fails, Section selection falls back to the head+tail policy from #3. See §6.

## 1. What the SEC requires: the structure we can rely on

- **10-K items.** Form 10-K (SEC 1673, rev. 02-25) has these Parts and Items:
  - Part I: Items 1 Business, 1A Risk Factors, 1B, 1C Cybersecurity, 2, 3, 4.
  - Part II: Items 5, 6 [Reserved], 7 MD&A, 7A Market Risk, 8 Financial Statements, 9, 9A Controls and Procedures, 9B, 9C.
  - Part III: Items 10–14.
  - Part IV: Items 15, 16.

  Source: [Form 10-K PDF](https://www.sec.gov/files/form10-k.pdf). Item 1A says "Smaller reporting companies are not required to provide the information required by this item", so a 1A that is missing or a one-liner is legitimate. Item 9A "Furnish[es] the information required by Item 307 and 308 of Regulation S-K", which covers controls and material weaknesses.
- **Captions are mandatory in principle.** Rule 12b-13 requires reports to contain "the numbers and captions of all items of the appropriate form", though the text of an item may be omitted if the answer shows what it covers ([17 CFR 240.12b-13](https://www.ecfr.gov/current/title-17/section-240.12b-13)). This is why "Item N." heading detection works at all.
- **Incorporation by reference breaks the one-document assumption.** Form 10-K General Instruction G:
  - G(2): Items 1 through 9A "may, at the registrant's option, be incorporated by reference from the registrant's annual report to security holders". Note 2 says that portion "shall be filed as an exhibit", which is Exhibit 13 under S-K Item 601(b)(13).
  - G(3): Part III may come from the proxy statement.
  - G(4): "No item numbers of captions of items need be contained in the material incorporated by reference". If everything is incorporated, the 10-K can consist of a "cross-reference sheet setting forth the item numbers and captions … and the page and/or pages in the referenced materials".
  - General Instruction H allows "integrated reports" that combine the annual report and the 10-K with a cross-reference sheet.

  These rules explain the WFC, GE, Citi and JPM shapes in §3.
- **10-Q** (SEC 1296, rev. 02-25) has Part I (Financial Information): Items 1 Financial Statements, 2 MD&A, 3 Market Risk, 4 Controls. It has Part II (Other Information): Items 1 Legal Proceedings, 1A Risk Factors, 2, 3, 4, 5, 6 Exhibits. Item numbers **repeat across Parts**, so the Part must be part of the key. Part II items that are "inapplicable or to which the answer is negative may be omitted and no reference thereto need be made". General Instruction D allows Part I to be incorporated by reference from a published quarterly report filed as an exhibit. Source: [Form 10-Q PDF](https://www.sec.gov/files/form10-q.pdf).
- **20-F** (SEC 1852, rev. 07-24) has Items 1–19. Risk factors are Item 3.D, the business is Item 4, the operating and financial review (the MD&A analogue) is Item 5, and the financial statements are Items 17/18. Cybersecurity is Item 16K. General Instruction (d) allows answering an item "by providing a cross-reference to the location of the information in the financial statements". Source: [Form 20-F PDF](https://www.sec.gov/files/form20-f.pdf).

## 2. Inline XBRL and EDGAR APIs as structure signals

- Measured on the 20 test filings (script: count of tags in the primary HTML):
  - 19/20 are inline XBRL.
  - **18/20 contain zero `<h1>`–`<h6>` tags.** One has 41 tags, but they mark cover-page fragments and notes, not Items. One has a single tag.
  - Headings are styled `<div>`/`<span>`/table cells, plus internal `href="#…"` links from the TOC: 0–499 per filing, and Wells Fargo has 0.
- iXBRL tags **facts**, not document structure. The useful exceptions:
  - `dei:DocumentType` gives the form (10-K, 10-Q, 20-F) for free. This is ground truth for the "document type" test Taxonomy and for picking a Section schema.
  - `us-gaap:…TextBlock` `ix:nonNumeric` facts wrap financial-statement notes and policies: 0–178 per filing.
  - The **Cybersecurity Disclosure (CYD) taxonomy**, supported since EDGAR 24.3 (Sept 2024), tags Item 1C / 20-F Item 16K disclosures one year after the disclosure compliance date ([SEC announcement](https://www.sec.gov/newsroom/whats-new/2409-2024-cyd-taxonomy), [CYD taxonomy guide](https://xbrl.sec.gov/cyd/2024/cyd-taxonomy-guide-2024-09-16.pdf)). 10 of the 13 10-Ks here carry `cyd:` tags.

  None of these mark 1A, 7 or 8 boundaries.
- **Fair access.** The SEC caps automated access at "10 requests/second" and requires you to "declare your user agent in request headers". Its sample format is `Sample Company Name AdminContact@<sample company domain>.com` ([Accessing EDGAR data](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)). **Empirical check: a request to `data.sec.gov/submissions/…` with `User-Agent: acceleread-research https://github.com/Jarrod-Bob/acceleread` (no email) returned HTTP 403 "Your Request Originates from an Undeclared Automated Tool".** I stopped fetching from SEC at that point. The filings below come instead from edgartools' MIT-licensed test fixtures, which are real EDGAR primary documents, and from the EDGAR-CORPUS dataset. **Implication for acceleread:** any EDGAR fetcher must take a user-supplied contact identity, including an email, and enforce ≤10 req/s. acceleread cannot ship a working default.

## 3. Experiment: 20 real filings, four detection approaches

**Corpus:** primary HTML documents from [edgartools `tests/fixtures/html`](https://github.com/dgunning/edgartools/tree/main/tests/fixtures/html):

- 10-Ks: AAPL, MSFT, NVDA, JPM, C, GE, WFC, XOM, PFE, ONDS, TALO, CIK 915358
- 10-Qs: AAPL, JPM, KO, TSLA, GS
- 20-Fs: three, one of which is not iXBRL

Ground truth was set by reading the first ~90 characters of each detected Section and checking plausibility.

### 3a. Docling HTML backend (docling 2.133.0)

- Docling's `html_backend.py` creates section headers only from `h1`–`h6` tags (`find_all(["h1", …, "h6"])` and `tag_name in {"h1", …}`). Bold `<b>`/`<strong>` only sets formatting, and CSS `font-weight` is ignored.
- Results:
  - AAPL 10-K: 1,075 text items, **0 headers**, 1.1 s.
  - MSFT 10-K: 1,885 text items, **0 headers**, 5.7 s.
  - 915358 10-K: 41 headers, none of them an Item. The Items are plain `text` paragraphs.
- The Item lines do survive as paragraphs, so a regex can run over Docling's text. But Docling adds nothing over a plain lxml text dump here. It also introduces letter-spacing artefacts ("RIS K FACTORS", "DESCRIP TION"), which a regex must tolerate.
- **Verdict:** don't use Docling for EDGAR HTML Sections.

### 3b. Naive regex baseline (about 60 lines, `lxml`, ~0.0–0.5 s/filing)

- **Method:**
  - Convert block elements to lines.
  - Match `^(PART X)? ITEM \d+[A-K]?[.:—]? caption$` on short lines.
  - Drop TOC rows, meaning dense clusters or rows ending in a page number.
  - Take the first remaining occurrence per (Part, Item).
- **Results:**
  - Agreed with a verified-correct Section (±10% chars) in **21/30** key Sections (Items 1/1A/7 and 10-Q MD&A).
  - Found **nothing** on Citi and GE (cross-reference index layouts).
  - Failure modes:
    - A cross-reference sentence or a running header starting a line ("Item 1" got 1.3k chars on PFE and JPM).
    - Repeated or split headings (ONDS 1A: 8.9k vs 122k).
    - JPM's 10-Q MD&A without Item captions at the start.

### 3c. edgartools 5.60.0 `edgar.documents.parse_html(html, ParserConfig(form=…))` (offline, MIT)

- This is a hybrid detector with three stages, used in order: TOC anchors (confidence 0.95), heading detection (0.7–0.9), regex patterns (0.6–0.7). It also reads cross-reference indexes. Every Section carries `confidence`, `detection_method`, `part`, `item`, `covered_items` (combined "Items 1 and 2") and size-band `warnings`. Runtime was 0.4–5.3 s per filing on one core.
- 10-K Items 1/1A/7, 13 filings (39 Sections):

  | Outcome | Count | Filings / Sections |
  | --- | --- | --- |
  | Clearly correct | ~31 | AAPL, MSFT, NVDA, ONDS, PFE, TALO, 915358, JPM all ✓; XOM 1/1A; GE 1A/7 |
  | Faithful but only a pointer, by design | 3 | WFC 1A/7 at 289–385 chars: "Information in response to this item can be found in…" the Exhibit 13 annual report |
  | Wrong | 2 | GE Item 1 returned the "Form 10-K cross reference index" page. XOM Item 7 started at the Financial Section TOC and overran |
  | Uncertain | 3 | Citi, which is parsed from its cross-reference index. 1A looks right, but 7A and 8 are 0.9–1.0 M chars, which is clearly wrong |
- Other forms:
  - 10-Q Part I Item 2 (MD&A): 5/5 plausible.
  - 20-F Items 3/4/5: plausible on 2 of 3. The non-iXBRL one returned 436 chars for Item 5, against 9.3k from the regex. **Unverified.**
- edgartools' own measured corpus agrees. Its README for `tests/fixtures/parser_corpus` says: "As of the current corpus, **17 of 54 filings (~31%)** carry a marker… Notable: `wfc` 10-K detects only 1 section, `c` (Citi) detects 0, and `gbdc`/`bac`/`ms` over-extract Item 8". It describes silent wrong content as "the worst failure class", for example GS `.business` at 668 KB and Citi at 1.78 MB of raw HTML. That is why it added size bands and confidence downgrades ([`section_size_bands.py`](https://github.com/dgunning/edgartools/blob/main/edgar/documents/section_size_bands.py), [corpus README](https://github.com/dgunning/edgartools/blob/main/tests/fixtures/parser_corpus/README.md)). My run shows parts of that are fixed in 5.60: WFC now yields 23 Sections, Citi 11. It also shows the large financial institutions are still the weak spot.
- **Conclusion.** For standard-layout filers, which is most operating companies, Item detection is reliable: 21/21 key Sections were correct for the seven standard-layout 10-Ks in this small sample. For banks and conglomerates that use integrated or cross-referenced annual reports, expect **wrong or partial Sections in a sizeable minority**. Confidence and size flags catch some of these, but not all.

### 3d. PDF (EDGAR 10-K HTML printed to PDF with headless Chrome)

| Filing | Pages | Docling PDF, OCR and tables off | Regex over pypdfium2 text | Regex over Docling markdown |
| - | - | - | - | - |
| AAPL 10-K | 78 | 39.8 s (including model load); **23/23 Item headings labelled `section_header`** | Item 7 = 229 chars (wrong: a wrapped "Item 7" cross-reference started a line), so 7A absorbed MD&A | 1/1A/7 all within ±10% of HTML |
| NVDA 10-K | 118 | 25.2 s; **23/23** | 1/1A/7 correct | 1/1A/7 correct |

- Docling's layout model sees the visual headings that the HTML markup hides.
- Requiring that an Item match is a `section_header` (rather than any line) removes the line-wrap false positive.
- **Caveat:** Chrome-printed EDGAR HTML is far cleaner than real PDF annual reports. Non-US glossy reports (UK/EU annual reports, PDF-only 20-F exhibits) have no Item numbers at all. Their structure is "Strategic report / Principal risks / Directors' report…", and they often carry a PDF outline (bookmarks). This experiment did not test them.

## 4. Open-source filing parsers (status as of 2026-10-03)

| Project | Licence | Latest release / last push | Forms | Notes |
| - | - | - | - | - |
| [edgartools](https://github.com/dgunning/edgartools) | MIT | v5.60.0, 2026-10-02 / pushed 2026-10-03; ~2.8k★ | 10-K, 10-Q, 8-K, 20-F (+ XBRL, many others) | Most capable and actively maintained. Offline `parse_html` works on HTML we already hold. Heavy core deps (pandas, pyarrow, rich, httpx, rapidfuzz, rank-bm25…). Very fast release cadence with module churn (`edgar.files` is deprecated in favour of `edgar.documents`), so **pin exactly**. Its downloader needs `set_identity("name email")` |
| [sec-parser](https://github.com/alphanome-ai/sec-parser) (alphanome-ai) | MIT | PyPI 0.58.1, 2024-06-09 / pushed 2026-06 | 10-Q-first; 10-K etc. "refer to this [exploration] document" | "Beta" badge. Semantic element tree, not Item segmentation. PyPI release is stale |
| [sec-parsers](https://github.com/john-friedman/SEC-Parsers) | MIT | PyPI 0.549, 2024-07-29 | 10-K, 10-Q, 8-K, S-1, 20-F | README: "**This package is no longer maintained**", pointing to datamule |
| [datamule](https://github.com/john-friedman/datamule-python) / [doc2dict](https://github.com/john-friedman/doc2dict) | MIT | datamule 5.0.2, 2026-07-27; doc2dict 0.7.1, 2026-02-03 | Many | Partly tied to paid hosted products ("SEC archive … $1/100k downloads"). The README admits "the docs are incomplete" |
| [edgar-crawler](https://github.com/lefterisloukas/edgar-crawler) | **GPL-3.0** | no releases / pushed 2025-07 | 10-K, 10-Q, 8-K items to JSON | Copyleft, so it is unsuitable as a dependency of an MIT/Apache library. Its output dataset [EDGAR-CORPUS](https://huggingface.co/datasets/eloukas/edgar-corpus) is Apache-2.0 and useful for evaluation |
| [sec-api Extractor API](https://sec-api.io/docs/sec-filings-item-extraction-api) | client MIT; **service paid** | sec-api 1.0.36, 2026-04-13 | 10-K items 1–15, 10-Q Part 1/2 items, 8-K items; no 20-F mentioned | Hosted, so it means sending documents to a third party. Docs recommend "focusing on filings post-2004" and note about "1 in 1,000 filings" have merged sections it can't split. Possible Escalation path only, not core |

## 5. Token sizes vs Jev's ~30k state budget

Two independent samples were measured with `chars / 3.5`. Text includes flattened tables, which inflates financial Sections.

**EDGAR-CORPUS, FY2020 10-Ks.** These are the first 214 filings of the test split: a broad mix of filer sizes, Sections split by edgar-crawler. "n" counts filings where the Section has more than about 1k chars. Filings below that are missing, "not applicable", or by-reference.

| Section | n / 214 | p10 | p50 | p90 | max | ≤30k tokens |
| - | - | - | - | - | - | - |
| Item 1 Business | 177 | 2.6k | **11.3k** | 31.2k | 81k | **89%** |
| Item 1A Risk Factors | 162 | 8.6k | **19.8k** | 48.1k | 92k | **73%** |
| Item 7 MD&A | 183 | 4.6k | **13.2k** | 28.6k | 99k | **93%** |
| Item 7A Market Risk | 95 | 0.4k | 1.2k | 2.8k | 14k | 100% |
| Item 8 Financial Statements | 117 | 10.5k | **26.9k** | 51.1k | 176k | 56% |
| Item 9A Controls | 184 | 0.6k | **1.2k** | 2.3k | 4k | 100% |
| **Whole 10-K** (all Sections) | 214 | 5.2k | **78.3k** | 140k | 273k | **17%** |

**edgartools size bands (24 large-cap 10-Ks, chars → tokens).** p50 / max:

- Item 1: 11.5k / 57.6k
- Item 1A: 22.8k / 57.6k
- Item 7: 16.3k / 46.6k
- Item 8: 37.3k / 81.4k
- Item 9A: 1.1k / 2.7k
- 10-Q Part I Item 2 MD&A: 14.5k / 28.7k

Source: [`size_bands.json`](https://github.com/dgunning/edgartools/blob/main/tests/fixtures/parser_corpus/size_bands.json).

**This experiment (verified Sections, tokens):**

| Filing | Whole document | Section sizes |
| --- | --- | --- |
| AAPL 10-K | 62k | 1 = 4.5k, 1A = 19.7k, 7 = 4.4k |
| MSFT 10-K | 98k | 1 = 13.9k, 1A = 19.7k, 7 = 13.7k |
| PFE 10-K | 213k | 1 = 25.3k, 7 = 30.3k |
| JPM 10-K | 407k | 1A = 39k, 7 = 125k |
| AAPL 10-Q | 20k | MD&A = 5.7k |
| KO 10-Q | 70k | MD&A = 22.2k |
| JPM 10-Q | 235k | MD&A = 77k |
| 20-F (two filers) | — | Items 3/4/5 = 4–10k each |

**What this means:**

- **Sector classification:** Item 1 alone fits about 89–90% of the time, and TypeSafe's own SEC cookbook trims to Item 1 for exactly this reason (see #4 and #3).
- **Questions:**
  - "Material weakness" belongs on Item 9A, which is about 1k tokens.
  - "MD&A outlook tone" belongs on Item 7 / 10-Q Part I Item 2, which fits about 93% of the time but not for big banks.
  - "Going-concern doubt" is usually discussed in MD&A/risk factors and in the auditor's report inside Item 8, the part that overflows most often.
- Sending the whole 10-K overflows about 83% of the time. **Section targeting is what makes filings fit without truncation**, and it also reduces the context rot noted in the long-documents research.

## 6. Recommended v0 approach

1. **`SectionDetector` seam**, chosen per Document by detected form. For EDGAR HTML, read the form from `dei:DocumentType` or the user-supplied metadata.
   - **EDGAR HTML 10-K/10-Q/20-F:** edgartools `edgar.documents.parse_html` (offline, MIT), exactly pinned, in an optional `acceleread[edgar]` extra so core stays light. Map its `Section` to ours, keeping `confidence`, `detection_method` and `warnings`.
   - **PDF (quality mode):** Docling `section_header` items. A Section starts at a header that matches the Item regex (`(PART X)? ITEM n[A-K]`), or at any header for generic Documents. Use the PDF outline (bookmarks) when present.
   - **Fast mode / anything else:** no Sections, so the Document is handled as a whole with head+tail truncation per #3. Don't run a line regex over pypdfium2 text: the false-positive rate is too high.
2. **Never trust a Section blindly.** Downgrade confidence when any of these hold:
   - Size is outside a per-(form, item) band.
   - The Section is a pointer only: short text matching `incorporated by reference|can be found|refer to|see .* (Annual Report|Exhibit 13|proxy)`.
   - Items are missing or duplicated.

   A low-confidence or absent target Section falls back to whole-Document head+tail. The Record notes which Section(s) were actually sent.
3. **Incorporation by reference.** v0 records `cross_reference=True` and the referenced location if detectable. It does **not** fetch Exhibit 13 or the proxy. Ingesting the EX-13 exhibit as part of the same Document is a v1 option, and requires the filing index and a fair-access fetcher with a user-supplied identity.
4. **Don't use** Docling for EDGAR HTML Sections, GPL edgar-crawler as a dependency, or the unmaintained sec-parsers. sec-api's hosted extractor is at most an opt-in Escalation.

### The generic Section abstraction this implies

```text
Section
  key            # stable canonical id, e.g. "risk_factors", "mdna", "business", "financial_statements", "controls"; or "heading:<slug>" for generic docs
  form_ref       # source-form locator, e.g. {"form": "10-Q", "part": "II", "item": "1A"} or None for non-filings
  title          # caption as found ("Item 1A. Risk Factors")
  span           # char offsets into the Document Record text (+ page range for PDFs); a Section may have >1 span (cross-reference index layouts)
  covered_items  # combined headings ("Items 1 and 2") -> ("1", "2")
  method         # toc | heading | pattern | cross_ref_index | layout | outline
  confidence     # 0..1
  flags          # cross_reference, undersized, oversized, missing
  est_tokens     # ceil(chars / 3.5)
```

- A **Section Schema** per form maps form items to canonical keys:

  | Canonical key | 10-K | 10-Q | 20-F |
  | --- | --- | --- | --- |
  | `mdna` | Item 7 | Part I Item 2 | Item 5 |
  | `risk_factors` | Item 1A | Part II Item 1A | Item 3.D |

  A Job or Question can then name a canonical key ("send `business` for sector; `controls` for the material-weakness Question") without caring about the form.
- Documents with no Schema still get generic heading Sections, or none.
- Sections are **optional metadata on a Document Record, never required**. Classification must work with zero Sections.

## 7. Newly surfaced questions

- **One Jev request per Document vs per-Question Section targeting.** The Jev integration decision (#4) assumes one request per Document with all Questions sharing one state. Section targeting wants different states for different Questions: Item 1 for sector, 9A for material weakness, 7 for tone. Options:
  - Compose a single state from several small Sections within 30k (e.g. 1 + 7 + 9A).
  - Group Questions by target Section, at N requests per Document.

  This needs a decision, plus evaluation data on accuracy vs cost.
- **Exhibit 13 / integrated annual reports.** Should a "Document" for an EDGAR filing be the primary document only, or the primary document plus the EX-13 annual report? This matters for WFC-style filers and affects the Document/Job model.
- **Real PDF annual reports (non-US):** heading and outline reliability, and how to map free-form headings ("Principal risks and uncertainties") to canonical keys. This needs a small corpus in the evaluation harness. It was not tested here.
- **Fair-access identity.** Any EDGAR fetch feature needs a user-supplied contact email in the User-Agent. Decide whether acceleread fetches from EDGAR at all in v0, or only ingests files the user provides.
- **Evaluation harness:** a labelled Section-boundary set (EDGAR-CORPUS, Apache-2.0, can bootstrap 10-K labels), and whether Item 1-only classification beats whole-text head+tail for sector.
- **Dependency weight:** edgartools pulls pandas and pyarrow. Is an optional extra acceptable, or should we vendor/port only the detector (MIT allows it, at a maintenance cost)?
