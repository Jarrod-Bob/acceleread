# PROTOTYPE — throwaway. Builds mock Document Records for the UI prototype from corpus/manifest.csv.
import csv, json, random, re
random.seed(7)
SECTORS = ["communication","consumer-discretionary","consumer-staples","energy","financials","health-care",
           "industrials","materials","real-estate","technology","utilities","other"]
DOCTYPE = {"10-K":"annual-report","10-Q":"quarterly-report","8-K":"current-report"}
rows = list(csv.DictReader(open("corpus/manifest.csv")))
recs = []
for i, r in enumerate(rows):
    company = re.sub(r"\s+\(.*$", "", r["company"]).strip()
    shell = r["going_concern"] == "likely"
    scan = "synthetic scan" in r["notes"]
    truth = r["sector"]
    # first-pass sector distribution
    if shell:
        top = random.choice([truth, "other", "other"])
        conf = round(random.uniform(0.38, 0.62), 2)
    else:
        top = truth if random.random() > 0.06 else random.choice(SECTORS[:-1])
        conf = round(random.uniform(0.86, 1.0), 2) if top == truth else round(random.uniform(0.45, 0.7), 2)
    rest = [s for s in SECTORS if s != top]
    runner = random.choice(rest)
    probs = {top: conf, runner: round(1 - conf - 0.01, 2), random.choice([s for s in rest if s != runner]): 0.01}
    gc = round(random.uniform(0.7, 0.97), 2) if shell else round(random.uniform(0.0, 0.06), 2)
    mw = round(random.uniform(0.4, 0.9), 2) if shell else round(random.uniform(0.0, 0.08), 2)
    outlook = round(random.uniform(0.2, 1.2), 2) if shell else round(random.uniform(1.3, 3.6), 2)
    reads_mdna = r["form"] in ("10-K", "10-Q")
    esc = {"status": "none"}
    if conf < 0.7:
        esc = {"status": "flagged", "reason": f"sector confidence {conf} < escalate_below 0.70"}
    status = "done"
    if i in (17, 41):
        status, esc = "failed", {"status": "none"}
    recs.append({
        "id": f"doc_{i:03d}", "file": r["file"], "company": company, "form": r["form"], "filed": r["filed"],
        "format": r["format"], "status": status, "attempts": 2 if status == "failed" else 1,
        "error": ("OCR worker crashed (BrokenProcessPool)" if i == 17 else "HTML parse error: no <body>") if status == "failed" else None,
        "truth_sector": truth,
        "classification": None if status == "failed" else {"category": top, "confidence": conf, "probabilities": probs,
            "coverage": "head+tail (truncated)" if not reads_mdna else "whole document" if r["form"] == "8-K" else "head+tail (truncated)"},
        "answers": None if status == "failed" else {
            "doc_type": {"kind": "choice", "value": DOCTYPE[r["form"]], "confidence": round(random.uniform(0.93, 1.0), 2), "coverage": "head+tail"},
            "going_concern": {"kind": "noul", "value": gc, "coverage": "mdna, financial_statements" if reads_mdna else "skipped: Section mdna missing"},
            "material_weakness": {"kind": "noul", "value": mw if reads_mdna else None, "coverage": "controls" if reads_mdna else "skipped: Section controls missing"},
            "outlook": {"kind": "score", "value": outlook if reads_mdna else None, "confidence": round(random.uniform(0.6, 0.95), 2), "coverage": "mdna" if reads_mdna else "skipped: Section mdna missing"},
        },
        "escalation": esc,
        "pages_ocr": 15 if scan else 0, "tokens": random.randint(2400, 24000) if status == "done" else 0,
    })
# two flagged ones were escalated to the LLM within the 2% cap... (60 docs → cap 1; show one escalated, rest flagged-over-cap)
flagged = [r for r in recs if r["escalation"]["status"] == "flagged"]
if flagged:
    r = flagged[0]
    first = dict(r["classification"])
    r["escalation"] = {"status": "escalated", "reason": first and f"sector confidence {first['confidence']} < 0.70",
                       "first": {"classifier": "jev-1.13.0", **first}}
    r["classification"] = {"category": r["truth_sector"], "confidence": None, "probabilities": None,
                           "coverage": "whole document (untruncated, claude-opus-5-5)"}
for r in flagged[1:]:
    r["escalation"]["reason"] += " — over escalation_max (2%), flagged only"
jobs = [
    {"id": "job_0007", "name": "corpus-60 · sector + filing risk", "status": "running", "total": 60,
     "taxonomy": "sector (12 Categories)", "question_sets": ["filing-risk v3 (sha 9f2c…)"], "profile": "fast",
     "escalate_below": 0.70, "escalation_max": "2%", "started": "10:42", "eta": "~4 min"},
    {"id": "job_0008", "name": "q3-10Q batch · filing risk only", "status": "queued", "total": 1840,
     "taxonomy": None, "question_sets": ["filing-risk v3"], "profile": "fast", "position": 1},
    {"id": "job_0009", "name": "non-US annual reports (PDF)", "status": "queued", "total": 212,
     "taxonomy": "sector (12 Categories)", "question_sets": [], "profile": "quality", "position": 2},
    {"id": "job_0006", "name": "articles triage · week 40", "status": "done", "total": 84,
     "taxonomy": "reading-priority (4 Categories)", "question_sets": [], "profile": "fast"},
    {"id": "job_0005", "name": "8-K sweep Sept", "status": "cancelled", "total": 3100,
     "taxonomy": "doc-type (5 Categories)", "question_sets": [], "profile": "fast"},
]
open("prototype/ui/data.js", "w").write("// PROTOTYPE — generated by make_mock_data.py, wipe me\nwindow.MOCK = " +
    json.dumps({"records": recs, "jobs": jobs, "sectors": SECTORS}, indent=1) + ";\n")
print(len(recs), "records;", sum(r["escalation"]["status"] != "none" for r in recs), "flagged/escalated")
