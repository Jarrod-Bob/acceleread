# Torch-free core; Docling as an optional `quality` extra

The base install of acceleread is torch-free. The default `fast` Extraction Profile uses pypdfium2 for the text layer and Tesseract on Pages the OCR rule flags. Docling, the more accurate layout-aware extractor, ships only in the opt-in `acceleread[quality]` extra on CPU torch. We chose this because `pip install docling` pulls about 3.25 GB of CUDA torch, against about 250 MB without torch. At 100k-Document scale, extraction speed rather than Jev is the bottleneck. A reader might expect the better extractor as the default. It isn't, because the install size and per-Page cost of Docling buy accuracy that most Pages, with clean text layers, don't need.

## Considered Options

- **Docling on ONNX Runtime without torch:** it works only with two monkeypatches and `transformers<5`, has no tables (TableFormer is torch-only), and its layout model is slower on CPU (0.53 vs 0.38 s/page). Revisit when docling#3792 lands.

## Consequences

- The OCR rule and Section Verification must work in both Profiles, so neither may assume Docling is installed.
- Automatic per-Document upgrade to `quality` is deferred until the evaluation harness can show when it pays off.
- Details: [Can Docling run on ONNX Runtime without torch?](https://github.com/Jarrod-Bob/acceleread/issues/13), [Should Jobs default to quality or fast extraction?](https://github.com/Jarrod-Bob/acceleread/issues/11)
