# SPDX-License-Identifier: Apache-2.0
"""PDF and HTML Extraction: the OCR rule applied per Page, Tesseract, per-Page provenance."""

from pathlib import Path

import pytest

from acceleread.extract import OcrLanguageUnavailable, extract_html, extract_pdf
from acceleread.ocr_rule import Step3Outcome

FIXTURES = Path(__file__).parent / "fixtures"


def page_text(doc_text: str, start: int, end: int) -> str:
    return doc_text[start:end]


def test_born_digital_pages_keep_their_text_layer_with_provenance() -> None:
    doc = extract_pdf(FIXTURES / "sample.pdf")
    for page in doc.pages:
        assert page.method == "text-layer"
        assert page.engine == "pdfium"
        assert page.engine_version
        assert page.ocr_decision is not None
        assert page.ocr_decision.step == 4
        assert page.image_coverage == 0.0
    assert doc.ocr_pages == 0


def test_scanned_page_is_recognised_with_tesseract() -> None:
    doc = extract_pdf(FIXTURES / "scanned.pdf")
    (page,) = doc.pages
    assert page.method == "ocr-full"
    assert page.engine == "tesseract"
    assert page.engine_version and page.engine_version.startswith("5.")
    assert page.ocr_languages == ["en"]
    assert page.ocr_confidence is not None and 0.5 < page.ocr_confidence <= 1.0
    assert page.ocr_decision is not None and page.ocr_decision.step == 1
    assert page.image_coverage == pytest.approx(1.0)
    assert "polysilicon" in doc.text
    assert "Revenue grew" in doc.text
    assert doc.ocr_pages == 1
    assert doc.ocr_ms > 0


def test_only_the_scanned_page_of_a_mixed_document_is_ocred() -> None:
    doc = extract_pdf(FIXTURES / "mixed.pdf")
    born_digital, scanned = doc.pages
    assert born_digital.method == "text-layer"
    assert scanned.method == "ocr-full"
    assert doc.text[born_digital.start : born_digital.end].startswith("Northwind Solar designs")
    assert "polysilicon" in doc.text[scanned.start : scanned.end]
    assert scanned.start == born_digital.end + 2
    assert doc.ocr_pages == 1


def test_text_beside_a_large_image_is_kept_in_fast() -> None:
    doc = extract_pdf(FIXTURES / "text_beside_image.pdf")
    (page,) = doc.pages
    assert page.method == "text-layer"
    assert page.image_coverage == pytest.approx(600 / 792, abs=0.01)
    assert doc.text.startswith("Northwind Solar designs")


def test_vector_only_page_goes_to_ocr_and_may_yield_no_text() -> None:
    doc = extract_pdf(FIXTURES / "vector.pdf")
    (page,) = doc.pages
    assert page.method == "ocr-full"
    assert page.ocr_decision is not None
    assert page.ocr_decision.path_count == 30
    assert page.ocr_decision.step == 1
    assert doc.text == ""


def test_step_three_hook_can_send_a_clean_page_to_ocr() -> None:
    def always_ocr(text: str) -> Step3Outcome | None:
        return Step3Outcome(ocr=True, word_ratio=0.1, jev_real_words=None, jev_skipped=True)

    doc = extract_pdf(FIXTURES / "mixed.pdf", step3=always_ocr)
    assert [p.method for p in doc.pages] == ["ocr-full", "ocr-full"]
    assert doc.pages[0].ocr_decision is not None and doc.pages[0].ocr_decision.step == 3
    assert doc.pages[0].ocr_decision.jev_skipped is True


def test_a_missing_language_pack_is_an_error_only_when_a_page_needs_ocr() -> None:
    with pytest.raises(OcrLanguageUnavailable, match="de"):
        extract_pdf(FIXTURES / "scanned.pdf", ocr_languages=["de"])
    text_only = extract_pdf(FIXTURES / "sample.pdf", ocr_languages=["de"])
    assert text_only.text.startswith("NORTHWIND SOLAR")


def test_page_counts_are_reported_once_the_ocr_rule_has_run() -> None:
    seen: list[tuple[int, int]] = []
    extract_pdf(FIXTURES / "mixed.pdf", on_page_counts=lambda pages, ocr: seen.append((pages, ocr)))
    assert seen == [(2, 1)]


def test_html_plain_extraction_drops_markup_scripts_and_styles(tmp_path: Path) -> None:
    html = tmp_path / "page.html"
    html.write_text(
        "<html><head><title>Quarterly update</title><style>p{color:red}</style>"
        "<script>var x = 1;</script></head>"
        "<body><h1>Results</h1><p>Revenue grew <b>18%</b> this year.</p>"
        "<ul><li>First</li><li>Second</li></ul><!-- hidden --></body></html>",
        encoding="utf-8",
    )
    doc = extract_html(html)
    assert doc.title == "Quarterly update"
    assert doc.pages == []
    assert "var x" not in doc.text and "color:red" not in doc.text and "hidden" not in doc.text
    assert doc.text.splitlines()[0] == "Results"
    assert "Revenue grew 18% this year." in doc.text
    assert "First\nSecond" in doc.text


def test_html_decodes_non_utf8_bytes_using_its_declared_charset(tmp_path: Path) -> None:
    html = tmp_path / "latin.html"
    html.write_bytes(
        b'<html><head><meta charset="iso-8859-1"></head><body><p>caf\xe9</p></body></html>'
    )
    assert extract_html(html).text.strip() == "café"
