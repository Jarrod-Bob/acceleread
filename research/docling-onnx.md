# Can Docling run on ONNX Runtime without torch?

Research for [#13](https://github.com/Jarrod-Bob/acceleread/issues/13), part of the v0 map ([#1](https://github.com/Jarrod-Bob/acceleread/issues/1)). It follows up an open question in [research/ocr-engine.md](https://github.com/Jarrod-Bob/acceleread/blob/research/ocr-engine/research/ocr-engine.md). Checked 2026-10-03 against docling v2.133.0 (released 2026-10-03, `main` at [`0cd61e0`](https://github.com/docling-project/docling/tree/0cd61e0050a9ef68e5e10495b87e41d31acd79c9)) and docling-ibm-models v4.0.3. Empirical runs were done in throwaway venvs on an Apple M5 (macOS arm64, Python 3.12, CPU only, 8 threads).

## Answer

- **Partly, and not supported out of the box.** Layout (Heron) and RapidOCR both have ONNX Runtime engines. **TableFormer has no ONNX path, so table structure always needs torch.** Two code bugs plus a transformers incompatibility mean that a torch-free install of v2.133.0 crashes. It works only after two small monkeypatches and pinning `transformers<5`. With those in place, a torch-free run gave **byte-identical Markdown** to the torch run on the test paper (tables off).
- **Size:** on Linux x86_64 (cp312, compressed wheels) the torch-free set is about **250 MB**. Plain `pip install docling` is about **3.25 GB**, because PyPI torch pulls CUDA. Docling with CPU-only torch is somewhere in between: the CPU torch wheel alone is about 196 MB. Docling's own `models-onnxruntime` extra pulls **onnxruntime-gpu (247 MB) on Linux**, so it doubles the torch-free set rather than shrinking it.
- **Throughput:** the ONNX layout engine was **about 25–40% slower** than the torch (transformers) engine on CPU: 0.53 vs 0.38 s/page for layout only. Removing torch does not make Docling faster. It makes it smaller.
- **Implication for acceleread:** a torch-free "quality" profile is possible, but only by depending on docling internals that upstream has not stabilised (PR [#3792](https://github.com/docling-project/docling/pull/3792) is open, with changes requested). For v0, ship Docling with **CPU-only torch** (from the PyTorch CPU index) as the quality profile, and keep the pypdfium2 + Tesseract **Lite** profile torch-free. Revisit torch-free Docling when #3792 or an equivalent fix lands.

## Per component

| Stage | Default engine | ONNX Runtime path? | Needs torch? | Source |
|---|---|---|---|---|
| PDF text layer (docling-parse / pypdfium2) | C++ / PDFium | n/a | No | `format-pdf` extra in [pyproject.toml](https://github.com/docling-project/docling/blob/0cd61e0050a9ef68e5e10495b87e41d31acd79c9/pyproject.toml) |
| Layout, `layout_heron_default` | Transformers (`default_engine_type=TRANSFORMERS`) | **Yes**: `OnnxRuntimeObjectDetectionEngineOptions` loads `docling-project/docling-layout-heron-onnx` | Inference no. Preprocessing uses HF `AutoImageProcessor`, which **needs torch/torchvision on transformers 5.x** | [stage_model_specs.py L1011–1027](https://github.com/docling-project/docling/blob/0cd61e0050a9ef68e5e10495b87e41d31acd79c9/docling/datamodel/stage_model_specs.py#L1011-L1027), [hf_vision_base.py](https://github.com/docling-project/docling/blob/0cd61e0050a9ef68e5e10495b87e41d31acd79c9/docling/models/inference_engines/common/hf_vision_base.py) |
| Other layout presets (Heron-101, Egret M/L/XL) | Transformers | **No.** "Only Heron has an ONNX export" | Yes | same file, L1029. [model_catalog.md](https://github.com/docling-project/docling/blob/0cd61e0050a9ef68e5e10495b87e41d31acd79c9/docs/usage/model_catalog.md): "Only `docling-layout-heron` also supports ONNXRuntime" |
| Table structure (TableFormer v1/v2) | docling-ibm-models | **No** | **Yes**. docling-ibm-models hard-depends on `torch`, `torchvision`, `safetensors[torch]` | [docling-ibm-models pyproject](https://github.com/docling-project/docling-ibm-models/blob/759c41a34c1c5148dc66901ef97d8962f445a152/pyproject.toml). Its [CHANGELOG](https://github.com/docling-project/docling-ibm-models/blob/main/CHANGELOG.md) shows v1.4.0 (2024-10-03) "Migration from onnx to pytorch script" and v2.0.0 "release ... with only torch models". Upstream issue [#760](https://github.com/docling-project/docling/issues/760) ("[ONNX] I can not convert table forcast to onnx", about TableFormer) is still open |
| OCR, RapidOCR | `RapidOcrOptions(backend="onnxruntime")` | **Yes** (PP-OCRv6 ONNX models ship in the `rapidocr` wheel) | Model no. But Docling's `RapidOcrModel.__init__` calls `decide_device()`, which does a bare `import torch` | [rapid_ocr_model.py L494](https://github.com/docling-project/docling/blob/0cd61e0050a9ef68e5e10495b87e41d31acd79c9/docling/models/stages/ocr/rapid_ocr_model.py#L494), [accelerator_utils.py L23](https://github.com/docling-project/docling/blob/0cd61e0050a9ef68e5e10495b87e41d31acd79c9/docling/utils/accelerator_utils.py#L23) |
| Picture classifier | Transformers | Yes (`OnnxRuntimeImageClassificationEngineOptions`) | Not needed by acceleread (off) | [model_family_engines_example.py](https://github.com/docling-project/docling/blob/0cd61e0050a9ef68e5e10495b87e41d31acd79c9/docs/examples/model_family_engines_example.py) |
| Reading order | Rule-based, in docling | n/a | No. It now lives in `docling/models/postprocessing/reading_order_rb.py` | [readingorder_model.py](https://github.com/docling-project/docling/blob/0cd61e0050a9ef68e5e10495b87e41d31acd79c9/docling/models/stages/reading_order/readingorder_model.py) |

## What upstream says about "ONNX without torch"

- The `models-onnxruntime` extra only adds `onnxruntime` on macOS and **`onnxruntime-gpu` on Linux and Windows**. It does not add `transformers` ([pyproject.toml](https://github.com/docling-project/docling/blob/0cd61e0050a9ef68e5e10495b87e41d31acd79c9/pyproject.toml), `[project.optional-dependencies]`). The docs example for the ONNX engines installs `docling[onnxruntime]`, which is `standard` (torch included) **plus** ONNX Runtime ([model_family_engines_example.py](https://github.com/docling-project/docling/blob/0cd61e0050a9ef68e5e10495b87e41d31acd79c9/docs/examples/model_family_engines_example.py), line 13). Upstream presents ONNX Runtime as an extra engine to use alongside torch, not as a replacement for it.
- The packaging guide says `models-local` "is the heavy one (pulls torch)". It describes `models-onnxruntime` only as "ONNX runtime backends", and lists no torch-free PDF recipe that uses ML models ([slim-packaging.md](https://github.com/docling-project/docling/blob/0cd61e0050a9ef68e5e10495b87e41d31acd79c9/docling/.agents/skills/docling/references/slim-packaging.md)). The installation docs still say "The Docling models depend on the PyTorch library" ([installation.md](https://github.com/docling-project/docling/blob/0cd61e0050a9ef68e5e10495b87e41d31acd79c9/docs/getting_started/installation.md)).
- Torch-free import work is in progress:
  - [#3805](https://github.com/docling-project/docling/issues/3805) (closed 2026-07-22) fixed module-level torch imports in chart-extraction and reading-order.
  - [#3997](https://github.com/docling-project/docling/issues/3997) (open) covers the VLM/ASR options modules, which still import `transformers`→torch whenever transformers is installed.
  - [#3996](https://github.com/docling-project/docling/issues/3996) (open) asks for a stable standalone ONNX layout path.
  - **PR [#3792](https://github.com/docling-project/docling/pull/3792) "optional deps handling for ONNX runtime"** (open, last updated 2026-09-22) would fix exactly the blockers below: lazy imports, CPU fallback in `decide_device`, guarded engine registration, and a numpy image processor for RT-DETR. A maintainer requested changes: it "does not deliver its stated transformer-free ONNX contract yet". A second maintainer warned that "changing the image processor (also if it looks the same) has shown in the past issues with the AI predictions" and that docling-ibm-models should instead get optional dependencies.

## Empirical check (throwaway venvs, 2026-10-03)

Install: `docling-slim[convert-core,format-pdf,feat-ocr-rapidocr-onnx,models-onnxruntime]==2.133.0` plus `transformers`. Torch was not installed. The pipeline used `RapidOcrOptions(backend="onnxruntime")`, `LayoutObjectDetectionOptions.from_preset("layout_heron_default")` with `OnnxRuntimeObjectDetectionEngineOptions()`, `do_table_structure=False`, `do_picture_classification=False` and device CPU.

Each failure in order:

1. Without `convert-core`, importing `DocumentConverter` fails: `No module named 'scipy'` (`base_ocr_model.py` imports scipy at module level). Upstream [#4447](https://github.com/docling-project/docling/issues/4447) reports the same thing. Fix: add `convert-core`.
2. `RapidOcrModel.__init__` → `decide_device()` → `ModuleNotFoundError: No module named 'torch'`. The same function is also called by the ONNX layout engine's provider selection. **Workaround: monkeypatch `decide_device` to return `"cpu"`.**
3. ONNX layout init with transformers 5.18 fails with "resolves to ... `RTDetrImageProcessorPil`, torchvision: `RTDetrImageProcessor`. None of these classes could be imported. Missing optional dependencies: torchvision". In transformers 5.x the PIL RT-DETR processor is decorated `@requires(backends=("torch",))`. **Workaround: pin `transformers<5` (4.57.6 worked; it uses the slow numpy processor).**
4. Building the table-structure factory imports `table_structure_model_granite_vision.py`, which does a module-level `import torch`. This happens even with `do_table_structure=False` (`docling/models/plugins/defaults.py`, `table_structure_engines()`). **Workaround: monkeypatch `table_structure_engines` to register only `TableStructureModel`.**
5. After 1–4: **conversion succeeds** with `torch` neither installed nor imported.

Results on `tests/data/pdf/sources/2206.01062.pdf` (DocLayNet paper, 9 born-digital pages, from the docling repo). Timings are after a warm-up page, single runs, so treat them as indicative:

| Venv | Layout engine | Tables | OCR (PDF-aware regions) | s/page | Markdown |
|---|---|---|---|---|---|
| torch-free | ONNX | off | on | 1.57–1.61 | 46,590 chars |
| torch-free | ONNX | off | off | 0.53 | 46,590 chars |
| torch (CPU) | ONNX | off | on | 1.60–1.61 | identical to torch-free |
| torch (CPU) | Transformers | off | on | 1.28 | identical (`diff` empty) |
| torch (CPU) | Transformers | off | off | 0.38 | 46,590 chars |
| torch (CPU) | ONNX | **on** | on | 2.30 | 52,489 chars |
| torch (CPU) | Transformers | **on** | on | 2.04 | 52,489 chars |

Takeaways:

- ONNX layout is **not faster** on CPU here. It is about 0.15 s/page slower than transformers (both at 8 threads).
- On this born-digital paper, **OCR of bitmap regions (figures) cost about 1 s/page and added zero characters to the output**. For classification, `do_ocr` on born-digital Pages is mostly wasted time. This supports gating OCR per Page, which the Lite heuristic already does, instead of running it on every picture region.
- TableFormer adds about 0.7 s/page here, and its output is only table formatting. Classification would rarely need it.

Venv footprint on macOS arm64: torch-free 580 MB on disk vs 1.1 GB with torch (macOS torch is CPU-only, 532 MB).

### Linux x86_64 wheel sizes (compressed, cp312, `uv pip compile --python-platform x86_64-manylinux_2_28` + PyPI JSON sizes)

| Dependency set | Packages | Compressed wheels | Biggest items |
|---|---|---|---|
| Torch-free: `docling-slim[convert-core,format-pdf,feat-ocr-rapidocr-onnx]` + `onnxruntime` + `transformers<5` | 65 | **~250 MB** | opencv-python 74, scipy 35, rapidocr 27, onnxruntime 24 MB |
| Same + `models-onnxruntime` extra | 66 | ~497 MB | **onnxruntime-gpu 247 MB**, installed alongside onnxruntime (both ship the `onnxruntime` module) |
| `docling-slim[...,models-local]` (PDF subset, PyPI torch) | 100 | ~3.27 GB | torch 555, nvidia-cudnn 553, cublas 423, triton 248 MB … |
| `docling==2.133.0` (standard) | 122 | ~3.25 GB | same CUDA stack |
| CPU-only torch alternative | — | ~250 MB + **~196 MB** torch CPU wheel (`torch-2.14.1+cpu`, download.pytorch.org) + docling-ibm-models/accelerate | the route docling's install docs recommend for Linux CPU |

Installed (uncompressed) sizes are roughly 2–3× larger. A Docker image adds the base OS and the model weights: Heron ONNX, TableFormer, and PP-OCRv6, which are bundled in the rapidocr wheel.

## Recommendation

1. **v0 packaging:** acceleread's `quality` extra depends on `docling-slim[convert-core,format-pdf,feat-ocr-rapidocr-onnx,models-local]`, **not** the `docling` meta-package, which pulls Office/web/email/chunking/CLI extras acceleread does not need. Document (and set in the Dockerfile) the PyTorch CPU index (`--extra-index-url https://download.pytorch.org/whl/cpu`) so the default image avoids CUDA. A GPU image can come later.
2. **Never use the `models-onnxruntime` extra** on Linux CPU images, because it pulls onnxruntime-gpu. If ONNX layout is wanted, depend on plain `onnxruntime` directly.
3. **Keep the base install torch-free:** acceleread core + Lite (pypdfium2 + Tesseract) has no torch. Docling sits behind the `Extractor` seam as an optional extra, so `pip install acceleread` stays small.
4. **Do not ship the monkeypatched torch-free Docling in v0.** It relies on private module internals (`docling.utils.accelerator_utils`, `docling.models.plugins.defaults`) and on `transformers<5`, which will conflict with other packages over time. Track [#3792](https://github.com/docling-project/docling/pull/3792), [#3997](https://github.com/docling-project/docling/issues/3997) and [#3996](https://github.com/docling-project/docling/issues/3996). Once upstream supports it, a torch-free "quality-lite" profile (Heron ONNX + RapidOCR ONNX, no tables) becomes a cheap add-on, because its output matched the torch run.
5. **Default profile:** this research does not change the "fast by default" lean. Docling stays at about 1.3–2 s/page on CPU with OCR and tables on, whichever runtime it uses.

## Open questions surfaced

- **Is TableFormer worth its cost for classification?** It needs torch, adds about 0.7 s/page, and only changes table formatting. Should the quality profile default to `do_table_structure=False`? The evaluation harness should measure this.
- **OCR on picture regions of born-digital Pages** cost about 1 s/page for zero extra text in this sample. Should the Docling profile run OCR only on Pages the Lite heuristic flags as textless (`do_ocr` per Page or per Document), instead of on all bitmap regions?
- **Docker image size budget:** what is acceptable (for example, under 1 GB CPU image)? This decides whether Docling is in the default image or a separate `-quality` tag.
- **Upstream contribution:** is it worth contributing the two small fixes (torch-optional `decide_device`, lazy Granite Vision table import) upstream, to unblock a torch-free profile sooner?
