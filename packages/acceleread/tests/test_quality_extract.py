# SPDX-License-Identifier: Apache-2.0
"""The `quality` Extraction Profile: Docling with RapidOCR on the Pages the OCR rule flags (§4.1).

Three layers, so most of it runs in CI:
- core tests (no Docling): weights checks, languages, and `assemble`, the pure mapping from
  Docling items to Pages, offsets and headings;
- tests with a fake Docling converter (need the `[quality]` extra, no weights): converter reuse,
  OCR runs, fallbacks, and the mapping from a real `DoclingDocument`, including a recorded one;
- tests that run Docling for real (extra and weights): they skip unless the Workspace named by
  `ACCELEREAD_TEST_WORKSPACE` (default: the default Workspace) holds `acceleread models fetch`
  output.
"""

import hashlib
import importlib.util
import os
import socket
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest

from acceleread import quality, quality_models
from acceleread.extract import OcrLanguageUnavailable, PageLayer, extract_pdf
from acceleread.models import Page
from acceleread.quality import (
    DoclingItem,
    ModelsMissing,
    QualityUnavailable,
    assemble,
    extract_pdf_quality,
)
from acceleread.quality_models import ModelPin, models_dir, models_status
from acceleread.workspace import resolve_workspace_path

FIXTURES = Path(__file__).parent / "fixtures"
HAS_DOCLING = importlib.util.find_spec("docling") is not None
needs_docling = pytest.mark.skipif(not HAS_DOCLING, reason="the [quality] extra is not installed")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tiny_pins() -> tuple[ModelPin, ...]:
    """Small stand-ins for the real pins: the same names, with one tiny file each."""
    return tuple(
        ModelPin(
            name=name,
            label=name,
            source="rapidocr" if name == "ocr" else "hf",
            folder=f"{name}-dir",
            repo_id=None if name == "ocr" else f"org/{name}",
            revision="r",
            files={"w.bin": sha(name.encode())},
            optional=name == "tables",
        )
        for name in ("layout", "ocr", "tables")
    )


@pytest.fixture
def weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A Workspace holding tiny pinned stand-ins for the layout and OCR weights."""
    monkeypatch.setattr(quality_models, "PINS", tiny_pins())
    for pin in tiny_pins():
        if not pin.optional:
            folder = models_dir(tmp_path) / pin.folder
            folder.mkdir(parents=True)
            (folder / "w.bin").write_bytes(pin.name.encode())
    return tmp_path


# --- core: weights are always required, and verified -------------------------------------------


def test_without_a_workspace_the_models_are_missing_and_never_downloaded() -> None:
    with pytest.raises(ModelsMissing, match=r"acceleread models fetch"):
        extract_pdf_quality(FIXTURES / "sample.pdf")


def test_an_empty_workspace_fails_the_document_naming_models_fetch(tmp_path: Path) -> None:
    with pytest.raises(ModelsMissing, match=r"acceleread models fetch"):
        extract_pdf_quality(FIXTURES / "sample.pdf", workspace=tmp_path)


def test_offline_changes_nothing_about_missing_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACCELEREAD_OFFLINE", "1")
    with pytest.raises(ModelsMissing, match=r"acceleread models fetch"):
        extract_pdf_quality(FIXTURES / "sample.pdf", workspace=tmp_path)


def test_weights_that_fail_their_hash_are_never_loaded(weights: Path) -> None:
    (models_dir(weights) / "layout-dir" / "w.bin").write_bytes(b"trunc")
    with pytest.raises(ModelsMissing, match=r"layout.*acceleread models fetch"):
        extract_pdf_quality(FIXTURES / "sample.pdf", workspace=weights)


def test_tables_need_the_tableformer_weights(weights: Path) -> None:
    with pytest.raises(ModelsMissing, match=r"models fetch --tables"):
        extract_pdf_quality(FIXTURES / "sample.pdf", workspace=weights, tables=True)
    assert models_status(weights) == {"layout": True, "ocr": True, "tables": False}  # untouched


def test_without_docling_the_profile_says_which_extra_to_install(
    weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(quality, "docling_available", lambda: False)
    with pytest.raises(QualityUnavailable, match=r"acceleread\[quality\]"):
        extract_pdf_quality(FIXTURES / "sample.pdf", workspace=weights)


def test_a_language_the_ocr_family_cannot_read_fails_when_a_page_needs_ocr() -> None:
    with pytest.raises(OcrLanguageUnavailable, match="ja"):
        extract_pdf_quality(FIXTURES / "scanned.pdf", ocr_languages=("ja",))


# --- core: assemble ----------------------------------------------------------------------------


def layer(number: int, text: str = "", *, ocr: bool = False) -> PageLayer:
    page = Page(number=number, start=0, end=0, engine="pdfium", engine_version="pdfium-1")
    return PageLayer(page, text, ocr)


def build(layers: Sequence[PageLayer], items: Sequence[DoclingItem]) -> Any:
    return assemble(layers, items, languages=["en"], docling_version="d1", rapidocr_version="r1")


def test_assemble_places_pages_and_headings_by_offset() -> None:
    out = build(
        [layer(1), layer(2)],
        [
            DoclingItem(1, "Item 1. Business", 0),
            DoclingItem(1, "We make panels.", None),
            DoclingItem(2, "Item 7. MD&A", 1),
            DoclingItem(2, "Revenue grew.", None),
        ],
    )
    assert out.text == "Item 1. Business\nWe make panels.\n\nItem 7. MD&A\nRevenue grew."
    first, second = out.pages
    assert out.text[first.start : first.end] == "Item 1. Business\nWe make panels."
    assert out.text[second.start : second.end].startswith("Item 7.")
    assert [(h.text, h.level) for h in out.headings] == [
        ("Item 1. Business", 0),
        ("Item 7. MD&A", 1),
    ]
    for heading in out.headings:
        assert out.text.startswith(heading.text, heading.start)
    assert {p.engine for p in out.pages} == {"docling"}


def test_a_page_docling_returns_nothing_for_keeps_its_text_layer() -> None:
    out = build(
        [layer(1, "Layer text one"), layer(2, "Layer text two")],
        [DoclingItem(2, "Docling two", None)],
    )
    first, second = out.pages
    assert out.text[first.start : first.end] == "Layer text one"
    assert (first.engine, first.method) == ("pdfium", "text-layer")  # provenance says so
    assert (second.engine, out.text[second.start : second.end]) == ("docling", "Docling two")


def test_an_ocr_page_is_attributed_to_rapidocr_even_when_it_is_empty() -> None:
    out = build([layer(1, "old layer", ocr=True)], [])
    (page,) = out.pages
    assert (page.method, page.engine, page.engine_version) == ("ocr-full", "rapidocr", "r1")
    assert page.ocr_languages == ["en"]
    assert out.text == ""


# --- Docling documents, no weights -------------------------------------------------------------


def a_doc(pages: dict[int, list[tuple[str, bool]]]) -> Any:
    """A DoclingDocument with (text, is_heading) items on the given Pages."""
    from docling_core.types.doc import BoundingBox, DocItemLabel, DoclingDocument, ProvenanceItem

    doc = DoclingDocument(name="t")
    box = BoundingBox(l=0, t=0, r=1, b=1)
    for number, items in pages.items():
        for text, heading in items:
            prov = ProvenanceItem(page_no=number, bbox=box, charspan=(0, len(text)))
            if heading:
                doc.add_heading(text=text, level=1, prov=prov)
            else:
                doc.add_text(label=DocItemLabel.TEXT, text=text, prov=prov)
    return doc


@needs_docling
def test_docling_items_split_a_multi_page_item_by_its_character_spans() -> None:
    from docling_core.types.doc import BoundingBox, DocItemLabel, DoclingDocument, ProvenanceItem

    doc = DoclingDocument(name="t")
    box = BoundingBox(l=0, t=0, r=1, b=1)
    item = doc.add_text(
        label=DocItemLabel.TEXT,
        text="ends page one starts page two",
        prov=ProvenanceItem(page_no=1, bbox=box, charspan=(0, 13)),
    )
    item.prov.append(ProvenanceItem(page_no=2, bbox=box, charspan=(14, 29)))
    assert quality.docling_items(doc) == [
        DoclingItem(1, "ends page one", None),
        DoclingItem(2, "starts page two", None),
    ]


@needs_docling
def test_docling_items_read_headings_as_zero_based_levels() -> None:
    doc = a_doc({1: [("Item 1. Business", True), ("Body", False)]})
    assert quality.docling_items(doc) == [
        DoclingItem(1, "Item 1. Business", 0),
        DoclingItem(1, "Body", None),
    ]


@needs_docling
def test_a_recorded_docling_document_maps_to_headings_and_pages() -> None:
    """report.docling.json is Docling's output for report.pdf (see make_report_pdf.py)."""
    from docling_core.types.doc import DoclingDocument

    doc = DoclingDocument.load_from_json(FIXTURES / "report.docling.json")
    items = quality.docling_items(doc)
    out = build([layer(n) for n in (1, 2, 3)], items)
    assert [h.text for h in out.headings] == [
        "Item 1. Business",
        "Item 1A. Risk Factors",
        "Item 7. Management's Discussion and Analysis",
    ]
    for heading, page in zip(out.headings, out.pages, strict=True):
        assert page.start <= heading.start < page.end
        assert out.text.startswith(heading.text, heading.start)


class FakeConverters:
    """Stands in for Docling: records how converters are built and what they are asked."""

    def __init__(self, pages: dict[int, list[tuple[str, bool]]]) -> None:
        self.pages = pages
        self.built: list[tuple[bool, bool]] = []  # (ocr, tables) per converter built
        self.calls: list[tuple[bool, int, int]] = []  # (ocr, first, last)

    def new(self, options: Any) -> Any:
        self.built.append((bool(options.do_ocr), bool(options.do_table_structure)))
        ocr = bool(options.do_ocr)

        def convert(path: Path, page_range: tuple[int, int]) -> Any:
            first, last = page_range
            self.calls.append((ocr, first, last))
            wanted = {n: v for n, v in self.pages.items() if first <= n <= last}
            return SimpleNamespace(document=a_doc(wanted))

        return SimpleNamespace(convert=convert)


@pytest.fixture
def fake_docling(weights: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    quality.reset_converters()
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    def install(pages: dict[int, list[tuple[str, bool]]]) -> FakeConverters:
        fake = FakeConverters(pages)
        monkeypatch.setattr(quality, "_new_converter", fake.new)
        return fake

    yield install
    quality.reset_converters()


@needs_docling
def test_converters_are_built_once_and_reused_across_documents(
    weights: Path, fake_docling: Any
) -> None:
    fake = fake_docling({1: [("one", False)], 2: [("two", False)]})
    extract_pdf_quality(FIXTURES / "sample.pdf", workspace=weights)
    extract_pdf_quality(FIXTURES / "sample.pdf", workspace=weights)
    assert fake.built == [(False, False)]  # one converter for two Documents
    extract_pdf_quality(FIXTURES / "scanned.pdf", workspace=weights)  # OCR needs its own
    extract_pdf_quality(FIXTURES / "scanned.pdf", workspace=weights)
    assert fake.built == [(False, False), (True, False)]


@needs_docling
def test_a_tables_converter_is_separate(weights: Path, fake_docling: Any) -> None:
    fake = fake_docling({1: [("one", False)], 2: [("two", False)]})
    folder = models_dir(weights) / "tables-dir"
    folder.mkdir(parents=True)
    (folder / "w.bin").write_bytes(b"tables")
    extract_pdf_quality(FIXTURES / "sample.pdf", workspace=weights)
    extract_pdf_quality(FIXTURES / "sample.pdf", workspace=weights, tables=True)
    assert fake.built == [(False, False), (False, True)]


@needs_docling
def test_runs_of_pages_are_converted_with_ocr_only_where_the_rule_flags(
    weights: Path, fake_docling: Any
) -> None:
    fake = fake_docling({1: [("digital", False)], 2: [("scanned words", False)]})
    doc = extract_pdf_quality(FIXTURES / "mixed.pdf", workspace=weights)
    assert fake.calls == [(False, 1, 1), (True, 2, 2)]
    assert [(p.method, p.engine) for p in doc.pages] == [
        ("text-layer", "docling"),
        ("ocr-full", "rapidocr"),
    ]
    assert doc.ocr_pages == 1
    assert doc.ocr_ms >= 0


@needs_docling
def test_docling_returning_nothing_for_a_page_keeps_the_pdf_text_layer(
    weights: Path, fake_docling: Any
) -> None:
    fake_docling({2: [("Item 7. Management's Discussion and Analysis", False)]})
    doc = extract_pdf_quality(FIXTURES / "sample.pdf", workspace=weights)
    first, second = doc.pages
    assert (first.engine, first.method) == ("pdfium", "text-layer")
    assert "NORTHWIND SOLAR" in doc.text[first.start : first.end]
    assert second.engine == "docling"


@needs_docling
def test_extraction_does_not_touch_process_environment(
    weights: Path, fake_docling: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_docling({1: [("one", False)], 2: [("two", False)]})
    monkeypatch.setenv("ACCELEREAD_OFFLINE", "1")
    before = dict(os.environ)
    extract_pdf_quality(FIXTURES / "sample.pdf", workspace=weights)
    assert dict(os.environ) == before


@needs_docling
def test_tables_are_off_unless_asked_for() -> None:
    assert quality.pipeline_options(None, ("en",), ocr=False).do_table_structure is False
    assert quality.pipeline_options(None, ("en",), ocr=False, tables=True).do_table_structure


@needs_docling
def test_ocr_runs_the_whole_page_with_the_latin_family() -> None:
    options = quality.pipeline_options(None, ("en", "de"), ocr=True)
    assert options.do_ocr and options.ocr_options.mode == "full_page"
    assert options.ocr_options.backend == "onnxruntime"  # pinned, never `auto`
    assert options.ocr_options.lang == ["latin"]
    assert quality.pipeline_options(None, ("en",), ocr=False).do_ocr is False


# --- real Docling, with weights ----------------------------------------------------------------


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


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any attempt to open a connection: with the weights installed, none is needed."""

    def refuse(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("the quality Profile tried to use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


def test_born_digital_pages_come_from_docling_with_per_page_ranges(
    workspace: Path, no_network: None
) -> None:
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
    assert "HF_HUB_OFFLINE" not in os.environ


def test_section_headers_become_headings_at_their_text_offsets(workspace: Path) -> None:
    doc = extract_pdf_quality(FIXTURES / "report.pdf", workspace=workspace)
    assert [h.text for h in doc.headings] == [
        "Item 1. Business",
        "Item 1A. Risk Factors",
        "Item 7. Management's Discussion and Analysis",
    ]
    for heading in doc.headings:
        assert doc.text.startswith(heading.text, heading.start)
    page_two = doc.pages[1]
    assert page_two.start <= doc.headings[1].start < page_two.end


def test_ocr_runs_only_on_the_pages_the_rule_flags(workspace: Path, no_network: None) -> None:
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


def test_a_scanned_page_is_recognised_with_rapidocr(workspace: Path, no_network: None) -> None:
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
