# Permissively licensed dependencies only

acceleread depends only on permissively licensed code and model weights (MIT, BSD, Apache-2.0, CDLA-Permissive). That keeps acceleread itself free to ship as MIT or Apache-2.0, and usable inside commercial pipelines. Three tempting tools are therefore banned: **PyMuPDF** (AGPL), **OCRmyPDF** (it requires Ghostscript, which is AGPL), and **Marker/Surya weights** (OpenRAIL-M with a $5M revenue cap, some CC-BY-NC-SA). PDF text comes from pypdfium2 or Docling, and OCR from Tesseract 5 or RapidOCR.

## Consequences

- A new dependency's licence, and the licence of any model weights it downloads at runtime, must be checked before adoption. A weights licence can differ from the code's.
- acceleread itself is **Apache-2.0** (decided in [Is acceleread MIT or Apache-2.0?](https://github.com/Jarrod-Bob/acceleread/issues/20)). CI enforces this ADR with a licence allowlist over each extra's resolved environment, and model weights and vendored assets are listed by hand in `THIRD_PARTY_NOTICES.md`.
- Details: [Which PDF extraction and OCR engine should v0 use?](https://github.com/Jarrod-Bob/acceleread/issues/2)
