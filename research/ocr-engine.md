# Which PDF extraction and OCR engine should v0 use?

Research for [#2](https://github.com/Jarrod-Bob/acceleread/issues/2), part of the v0 map ([#1](https://github.com/Jarrod-Bob/acceleread/issues/1)). Checked 2026-10-03 against primary sources: GitHub repos, LICENSE/README files, Hugging Face model cards, vendor pricing pages, and published benchmarks. Versions and licenses come from the GitHub API on that date.

## Answer

- **Primary: [Docling](https://github.com/docling-project/docling)** (v2.133.0, code MIT). Pin its OCR engine to **RapidOCR on ONNX Runtime** instead of using `auto`. Docling already does what acceleread needs: it reads the text layer first and OCRs only the regions with no text (`OcrMode.DEFAULT` maps to `PDF_AWARE_LAYOUT_REGIONS`). It also handles reading order, multi-column layout and tables. Every default dependency is permissively licensed.
- **Fallback: a lightweight path built from [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) and [Tesseract 5](https://github.com/tesseract-ocr/tesseract)** (Apache-2.0/BSD-3 and Apache-2.0). pypdfium2 reads the text layer. Tesseract OCRs whole pages that have no usable text. This path has no layout model and no torch. Use it when Docling fails on a Document or Page, and as a high-throughput "fast" profile. For classification, the text matters more than its layout.
- **License implication:** with these choices acceleread can ship under **MIT or Apache-2.0**. Keep these out of the default dependency tree: **PyMuPDF** (AGPL-3.0), **Marker/Surya model weights** (modified OpenRAIL-M with a revenue cap, and some CC-BY-NC-SA-4.0), and a bundled **Ghostscript** (AGPL-3.0, which OCRmyPDF requires).

## Comparison

| Engine | Version (2026-10-03) | Code license | Model/weights license | Text-layer first? | Per-page / per-region OCR fallback | Layout / tables | Install weight | Throughput evidence |
|---|---|---|---|---|---|---|---|---|
| **Docling** | v2.133.0 | MIT | Layout "Heron": Apache-2.0. `docling-models` (TableFormer): CDLA-Permissive-2.0 / Apache-2.0 | Yes (docling-parse or pypdfium2 backends) | Yes, per region. Modes: `FULL_PAGE`, `LAYOUT_REGIONS`, `PDF_AWARE_LAYOUT_REGIONS` (default) | Yes (layout model, TableFormer, reading order) | Heavy: `standard` extra pulls torch + torchvision + rapidocr | Docling paper: 3.1 s/page x86 CPU, 1.27 s/page M3 Max, 0.49 s/page L4, all with OCR and tables on. Turning OCR off saves about 60% on CPU. Marker's README measures docling at 50.3 olmocr-bench, 2.1 pg/s on a B200 |
| **Marker** (+ Surya) | marker v2.0.0, surya v0.22.1 | Apache-2.0 | **Modified OpenRAIL-M**: free for research, personal use and startups under $5M funding/revenue, paid above that. Several Surya models on HF are **CC-BY-NC-SA-4.0** | Yes (pdftext) | Yes, selective OCR | Yes, strongest of the OSS pipelines in its own benchmark | Heavy (torch, plus a VLM inference server for OCR modes) | Vendor-run olmocr-bench on 1× B200: balanced 76.0 / 2.9 pg/s, fast 66.6 / 7.4 pg/s, no-OCR CPU 43.6 / 23.7 pg/s |
| **PaddleOCR** (PP-OCRv5/v6, PP-StructureV3, PaddleOCR-VL) | v3.7.0 | Apache-2.0 | Apache-2.0 (per repo) | PP-StructureV3 parses PDFs, but text-layer-first is not its main design | You would build it yourself | Yes (PP-StructureV3, PaddleOCR-VL 0.9B) | Heavy: PaddlePaddle framework. Models also run through RapidOCR/ONNX without Paddle | README: PP-OCRv6 claims a 5.2× CPU speedup (OpenVINO) and 0.13 s on A100. PaddleOCR-VL-1.6 claims 96.3% on OmniDocBench v1.6 (vendor) |
| **docTR** | v1.1.0 | Apache-2.0 | Apache-2.0 | No, it is OCR only | You would build it yourself | Detection + recognition only, no table structure | Medium (PyTorch) | No first-party PDF-throughput numbers found |
| **Tesseract** via **OCRmyPDF** | Tesseract 5.5.3, OCRmyPDF v17.13.0 | Tesseract Apache-2.0. OCRmyPDF MPL-2.0 | Tesseract `tessdata`: Apache-2.0 | OCRmyPDF can skip pages that already have text (`--skip-text`, `--redo-ocr`) | Per page | None. Output is a searchable PDF, not structured text | Light Python package, but needs external binaries: Tesseract plus **Ghostscript (AGPL-3.0)** | CPU only, no layout model. Speed depends on DPI and language |
| **Cloud: Mistral OCR 4.1** | API | Proprietary | n/a | n/a (renders every page) | n/a | Yes (Markdown + tables) | None locally | **$4 / 1000 pages** (Document AI $5 / 1000). Batch "at half price" |
| **Cloud: AWS Textract** | API | Proprietary | n/a | n/a | n/a | Tables: $15 / 1000 pages | None locally | Detect Document Text: **$1.50 / 1000 pages**. Analyze Document (Tables): **$15 / 1000 pages** (US West Oregon, first 1M pages) |

## Findings and sources

### Licensing

- **Docling** code is MIT. The README says "For individual model usage, please refer to the model licenses found in the original packages" ([README](https://github.com/docling-project/docling#license)). The default layout model `docling-project/docling-layout-heron` is Apache-2.0. `docling-project/docling-models`, which holds TableFormer, is tagged CDLA-Permissive-2.0 and Apache-2.0 ([HF API: heron](https://huggingface.co/api/models/docling-project/docling-layout-heron), [HF API: docling-models](https://huggingface.co/api/models/docling-project/docling-models)).
- **Marker** code moved to Apache-2.0, but "Our model weights use a modified AI Pubs Open Rail-M license (free for research, personal use, and startups under $5M funding/revenue)" ([marker README, "Commercial usage"](https://github.com/datalab-to/marker#commercial-usage)). On Hugging Face, `datalab-to/surya-ocr-2` and `surya_layout2` are tagged `openrail`. `surya_layout`, `surya_tablerec`, `ocr_error_detection` and `texify` are tagged **`cc-by-nc-sa-4.0`** ([HF API listing](https://huggingface.co/api/models?author=datalab-to)). acceleread is a library other people will run in team pipelines, so this is a liability: users above the cap would need a Datalab licence.
- **PaddleOCR** is Apache-2.0 ([README License section](https://github.com/PaddlePaddle/PaddleOCR#-license)). **docTR** is Apache-2.0 ([README](https://github.com/mindee/doctr#license)). **Tesseract** is Apache-2.0 (GitHub API). **RapidOCR** is Apache-2.0 (GitHub API, `rapidai/RapidOCR`).
- **OCRmyPDF** is MPL-2.0. MPL-2.0 "permits integration ... included commercial and closed source" ([README](https://github.com/ocrmypdf/OCRmyPDF#license)). But OCRmyPDF "requires external program installations of Ghostscript and Tesseract OCR" (same README). Artifex states Ghostscript and MuPDF are "dual-licensed under ... the GNU AGPLv3 license ... or with commercial license agreements" ([artifex.com/licensing](https://artifex.com/licensing)). Calling Ghostscript as a subprocess does not relicense acceleread. Shipping it in an acceleread Docker image does mean distributing AGPL software.
- **PyMuPDF** is **AGPL-3.0** (GitHub API, `pymupdf/PyMuPDF`). It is a common choice for text-layer extraction and should not be used. **pypdfium2** is "Apache-2.0 / BSD-3-Clause", and PDFium is BSD-style ([README "Licensing"](https://github.com/pypdfium2-team/pypdfium2#licensing)).

### Text-layer detection and per-page OCR fallback

- Docling: `PdfPipelineOptions.do_ocr` defaults to `True`, and `ocr_options` defaults to `OcrAutoOptions()`. `OcrMode.DEFAULT` "is wired to run PDF_AWARE_LAYOUT_REGIONS", which means "Eliminate those clusters that contain exclusively text PDF cells". So OCR runs only on layout regions that have no programmatic text. `FULL_PAGE` forces OCR on every page, and `force_full_page_ocr` is now deprecated in its favour. OCR renders at 72 DPI × `scale` (default 3, so 216 DPI) ([`docling/datamodel/pipeline_options.py`](https://github.com/docling-project/docling/blob/main/docling/datamodel/pipeline_options.py)).
- Docling `auto` OCR selection order ([`auto_ocr_model.py`](https://github.com/docling-project/docling/blob/main/docling/models/stages/ocr/auto_ocr_model.py)):
  1. ocrmac, on macOS
  2. RapidOCR with onnxruntime
  3. EasyOCR
  4. RapidOCR with torch

  The macOS result would differ from Linux, so **acceleread should pin `RapidOcrOptions(backend="onnxruntime")`** (or Tesseract) for reproducible records.
- Docling's OCR options are RapidOCR, EasyOCR, Tesseract (CLI or tesserocr), ocrmac, Nemotron OCR, and a KServe v2 remote endpoint (same file).
- Marker reads the text layer with pdftext and "OCRs selectively". Its `--disable_ocr` mode is "the pure CPU text-layer path" ([marker README, Benchmarks](https://github.com/datalab-to/marker#benchmarks)).
- docTR and PaddleOCR are OCR engines, not PDF pipelines. acceleread would have to write the text-layer detection itself.

### Accuracy on articles and papers (multi-column, tables)

- The only fresh head-to-head is **vendor-run** by Datalab. It uses AllenAI's third-party [olmocr-bench](https://github.com/allenai/olmocr/tree/main/olmocr/bench): 1,403 PDFs with categories for multi-column, tables, arXiv math and old scans. Overall scores: Marker balanced 76.0, MinerU pipeline 72.7, Marker fast 66.6, **docling 50.3**, Marker no-OCR 43.6. Born-digital-only: Marker balanced 83.5, docling 64.0 ([marker README](https://github.com/datalab-to/marker#benchmarks)). Treat the docling number with care: a competitor ran docling at default settings.
- PaddleOCR-VL-1.6 claims 96.3% on OmniDocBench v1.6 ([PaddleOCR README](https://github.com/PaddlePaddle/PaddleOCR)). This is a VLM doing full-page parsing, a different cost class from text-layer extraction.
- **For acceleread's use, extraction quality matters mainly as input to classification.** Jev classifies on text, and reading-order mistakes or table formatting rarely change a Category verdict. That is why a plain text-layer fallback is acceptable. The evaluation-harness ticket should confirm it on a real sample corpus.

### Throughput (CPU vs GPU)

- Docling technical report ([arXiv 2501.17887](https://arxiv.org/html/2501.17887)) used 89 PDFs, 4,008 pages, with OCR and table recognition on. Docling ran at 3.1 s/page on x86 CPU (8 threads), 1.27 s/page on M3 Max and 0.49 s/page on an L4 GPU. In the same run Marker took 16+ s/page on CPU. "Disabling OCR saves 60% of runtime on the x86 CPU and the M3 Max SoC." These figures come from early-2025 versions.
- Marker README, 1× B200 with concurrent workers: docling 2.1 pg/s, Marker balanced 2.9, fast 7.4, no-OCR CPU 23.7 pg/s.
- **Scale check:** a 100k-Document Job of articles (~10 pages each) is about 1M Pages. At around 1 s/page/core of CPU Docling with OCR off for born-digital pages, that is roughly 1M core-seconds, or about 1.5 days on 8 cores. A pure pypdfium2 text dump takes milliseconds per page. Jev does not limit throughput: 100K tok/s means 1M pages × ~500 tokens is about 1.4 h. **Extraction is the bottleneck**, so v0 needs the lightweight profile as a first-class option, not only as an error fallback.

### Cloud option

- Mistral OCR 4.1: $4 / 1000 pages, Document AI $5 / 1000, batch "at half price" ([mistral.ai/pricing/api](https://mistral.ai/pricing/api)). AWS Textract (US West Oregon, first 1M pages): Detect Document Text $1.50 / 1000, Analyze Document Tables $15 / 1000 ([aws.amazon.com/textract/pricing](https://aws.amazon.com/textract/pricing/)).
- The map settled "OCR stays local", so cloud OCR is out for v0. It could become an opt-in OCR backend behind the same seam, since Docling supports a remote KServe OCR endpoint. At $1.5–$4 per 1000 pages, sending only OCR-needing pages is cheap. The obstacle is privacy, not cost.

## Recommendation detail

1. **Extraction seam**: put an `Extractor` interface behind the Document Record so engines can be swapped. Two implementations:
   - `DoclingExtractor` (default, "quality" profile): `PdfPipelineOptions(do_ocr=True, ocr_options=RapidOcrOptions(backend="onnxruntime"))` with the default PDF-aware OCR mode. Tables on.
   - `LiteExtractor` ("fast" profile, and the fallback when Docling errors or times out on a Document): pypdfium2 text per Page, then a per-Page heuristic (character count, ratio of image area, share of garbage glyphs) that sends empty or garbled Pages to Tesseract 5 through `tesserocr` or the CLI.
2. **Record the method per Page** in page metadata (text-layer, OCR-region or OCR-full-page, plus engine and version). CONTEXT.md already makes Page "the unit at which extraction method is decided".
3. **Dependencies:** no PyMuPDF, no Marker/Surya weights, no bundled Ghostscript. Then acceleread can be MIT/Apache-2.0.

## Open questions surfaced

- **Docling without torch:** Docling now ships as `docling-slim` with extras. `standard` pulls `models-local` (torch + torchvision), and a separate `models-onnxruntime` extra exists ([pyproject.toml](https://github.com/docling-project/docling/blob/main/pyproject.toml)). It is unverified whether the layout and table models can run on ONNX Runtime alone. If they can, install weight and Docker image size drop sharply. This affects packaging.
- **Docling CPU throughput on current versions** has not been measured on our kind of corpus. The published numbers are from early-2025 versions, or were run by a competitor. This should go into the evaluation-harness ticket as a benchmark of Docling vs Lite profiles on a sample corpus, measuring pages/sec and the classification-accuracy delta.
- **Default profile:** is the default `Job` profile "quality" (Docling) or "fast" (Lite)? The 1M-page scale suggests fast by default, but this is a product decision.
- **Per-page OCR heuristic thresholds** for the Lite path, such as the minimum character count and how to detect garbled glyphs, need a small prototype on real PDFs.
- **Cloud OCR as an opt-in backend:** does "OCR stays local" forbid it entirely, or allow it as an explicit opt-in?
