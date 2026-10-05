# SPDX-License-Identifier: Apache-2.0
"""The `quality` Extraction Profile: Docling with RapidOCR on the Pages the OCR rule flags (§4.1).

Tests that run Docling need the `[quality]` extra and the model weights from `acceleread models
fetch`. They skip without them: point `ACCELEREAD_TEST_WORKSPACE` at a Workspace that has them,
otherwise the default Workspace is used. The rest run in the core install.
"""

import importlib.util
import os
from pathlib import Path

import pytest

from acceleread import quality
from acceleread.extract import OcrLanguageUnavailable, extract_pdf, profile_for
from acceleread.models import JobSettings
from acceleread.quality import ModelsMissing, QualityUnavailable, extract_pdf_quality
from acceleread.quality_models import models_status
from acceleread.workspace import resolve_workspace_path

FIXTURES = Path(__file__).parent / "fixtures"
HAS_DOCLING = importlib.util.find_spec("docling") is not None


@pytest.fixture(scope="module")
def workspace() -> Path:
    if not HAS_DOCLING:
        pytest.skip("the [quality] extra is not installed")
    configured = os.environ.get("ACCELEREAD_TEST_WORKSPACE")
    path = Path(configured) if configured else resolve_workspace_path(None)
    status = models_status(path)
    if not (status["layout"] and status["ocr"]):
        pytest.skip("run `acceleread models fetch` (or set ACCELEREAD_TEST_WORKSPACE)")
    return path


def test_without_docling_the_profile_says_which_extra_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quality, "docling_available", lambda: False)
    with pytest.raises(QualityUnavailable, match=r"acceleread\[quality\]"):
        extract_pdf_quality(FIXTURES / "sample.pdf")


def test_offline_without_the_models_fails_instead_of_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACCELEREAD_OFFLINE", "1")
    with pytest.raises(ModelsMissing, match="models fetch"):
        extract_pdf_quality(FIXTURES / "sample.pdf", workspace=tmp_path)


def test_a_language_the_ocr_family_cannot_read_fails_when_a_page_needs_ocr() -> None:
    with pytest.raises(OcrLanguageUnavailable, match="ja"):
        extract_pdf_quality(FIXTURES / "scanned.pdf", ocr_languages=("ja",))


def test_per_document_profile_override_beats_the_job_profile() -> None:
    job = JobSettings.model_validate(
        {
            "inputs": [{"source": "a.pdf"}, {"source": "b.pdf"}],
            "extraction_profile": "fast",
            "overrides": {"b.pdf": {"extraction_profile": "quality"}},
        }
    )
    assert profile_for(job, "a.pdf") == "fast"
    assert profile_for(job, "b.pdf") == "quality"
    assert profile_for(job, Path("b.pdf")) == "quality"


def test_tables_are_off_unless_asked_for() -> None:
    pytest.importorskip("docling")
    assert quality.pipeline_options(None, ("en",), ocr=False).do_table_structure is False
    assert quality.pipeline_options(None, ("en",), ocr=False, tables=True).do_table_structure


def test_ocr_runs_the_whole_page_with_the_latin_family() -> None:
    pytest.importorskip("docling")
    options = quality.pipeline_options(None, ("en", "de"), ocr=True)
    assert options.do_ocr and options.ocr_options.mode == "full_page"
    assert options.ocr_options.backend == "onnxruntime"  # pinned, never `auto`
    assert options.ocr_options.lang == ["latin"]
    assert quality.pipeline_options(None, ("en",), ocr=False).do_ocr is False


def test_born_digital_pages_come_from_docling_with_per_page_ranges(workspace: Path) -> None:
    doc = extract_pdf_quality(FIXTURES / "sample.pdf", workspace=workspace)
    first, second = doc.pages
    assert "Item 1. Business" in doc.text[first.start : first.end]
    assert "Item 7." in doc.text[second.start : second.end]
    for page in doc.pages:
        assert (page.method, page.engine) == ("text-layer", "docling")
        assert page.engine_version
        assert page.ocr_decision is not None and page.ocr_decision.step == 4
    assert doc.ocr_pages == 0
    assert doc.title == "Northwind Solar Annual Report"


def test_section_headers_become_headings_at_their_text_offsets(workspace: Path) -> None:
    doc = extract_pdf_quality(FIXTURES / "report.pdf", workspace=workspace)
    texts = [h.text for h in doc.headings]
    assert texts == [
        "Item 1. Business",
        "Item 1A. Risk Factors",
        "Item 7. Management's Discussion and Analysis",
    ]
    for heading in doc.headings:
        assert doc.text.startswith(heading.text, heading.start)
    page_two = doc.pages[1]
    assert page_two.start <= doc.headings[1].start < page_two.end


def test_ocr_runs_only_on_the_pages_the_rule_flags(workspace: Path) -> None:
    counts: list[tuple[int, int]] = []
    doc = extract_pdf_quality(
        FIXTURES / "mixed.pdf",
        workspace=workspace,
        on_page_counts=lambda pages, ocr: counts.append((pages, ocr)),
    )
    fast = extract_pdf(FIXTURES / "mixed.pdf")
    assert counts == [(2, 1)]
    assert doc.ocr_pages == 1 and doc.ocr_ms > 0
    flagged = [p for p in doc.pages if p.method == "ocr-full"]
    kept = [p for p in doc.pages if p.method == "text-layer"]
    assert len(flagged) == len(kept) == 1
    # The same Pages are flagged as in `fast`: the rule is shared.
    assert [p.ocr_decision for p in doc.pages] == [p.ocr_decision for p in fast.pages]
    assert flagged[0].engine == "rapidocr"
    assert flagged[0].ocr_languages == ["en"]
    assert kept[0].engine == "docling"


def test_a_scanned_page_is_recognised_with_rapidocr(workspace: Path) -> None:
    doc = extract_pdf_quality(FIXTURES / "scanned.pdf", workspace=workspace)
    (page,) = doc.pages
    assert page.method == "ocr-full" and page.engine == "rapidocr"
    assert "margin" in doc.text.lower()


def test_the_worker_handler_runs_the_profile_the_task_names(workspace: Path) -> None:
    from acceleread.workers import ExtractTask, extract_document

    task = ExtractTask(FIXTURES / "report.pdf", "pdf", profile="quality", workspace=workspace)
    doc = extract_document(task, lambda pages, ocr: None)
    assert {p.engine for p in doc.pages} == {"docling"}
    assert len(doc.headings) == 3
    fast = extract_document(ExtractTask(FIXTURES / "report.pdf", "pdf"), lambda p, o: None)
    assert {p.engine for p in fast.pages} == {"pdfium"}
    assert fast.headings == ()
