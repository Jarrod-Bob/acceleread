# SPDX-License-Identifier: Apache-2.0
"""The full Document Record schema (docs/spec/v0.md §6)."""

from typing import Any

from acceleread.models import SCHEMA_VERSION, DocumentRecord, record_json_schema

FULL: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "record_id": "r1",
    "job_id": "job_1",
    "external_id": "ext-9",
    "user_metadata": {"batch": "q3", "n": 4},
    "status": "partial",
    "errors": [{"stage": "classify", "code": "Over", "message": "m", "page": 2}],
    "attempts": 2,
    "source": {
        "filename": "a.pdf",
        "format": "pdf",
        "sha256": "ab",
        "bytes": 10,
        "group": "0000320193-24-000123",
    },
    "metadata": {
        "title": {"value": "Annual Report", "source": "pdf-info"},
        "cik": {"value": 320193, "source": "edgar-header"},
    },
    "extraction_profile": "quality",
    "text": "hello",
    "pages": [
        {
            "number": 1,
            "start": 0,
            "end": 5,
            "method": "ocr-full",
            "engine": "tesseract",
            "engine_version": "5.5.0",
            "ocr_languages": ["en"],
            "ocr_confidence": 0.91,
            "ocr_decision": {
                "step": 1,
                "reason": "image-covered",
                "chars": 3,
                "word_ratio": 0.0,
                "bad_char_ratio": 0.0,
                "image_coverage": 0.9,
                "path_count": 0,
                "jev_real_words": None,
                "jev_skipped": True,
            },
            "image_coverage": 0.9,
        }
    ],
    "sections": [
        {
            "keys": ["business", "risk_factors"],
            "label": "Items 1 and 1A",
            "form_ref": "10-K 1",
            "spans": [{"start": 0, "end": 5}],
            "method": "toc_anchor",
            "confidence": 0.95,
            "verification": {"status": "verified", "p": 0.97},
            "flags": ["pointer"],
            "est_tokens": 2,
        }
    ],
    "taxonomy": {"name": "t", "hash": "sha256:00"},
    "classification": {
        "kind": "choice",
        "value": "energy",
        "probabilities": {"energy": 0.9, "other": 0.1},
        "confidence": 0.9,
        "classifier": {"id": "jev", "model": "jev-1.13.0", "version": "1.13.0"},
        "coverage": {
            "sections": ["business"],
            "truncated": True,
            "shrunk": True,
            "est_tokens": 10,
            "input_tokens": 12,
            "note": "pointer → Exhibit 13",
        },
        "escalation": {
            "status": "escalated",
            "reason": "below 0.95",
            "first": {
                "classifier": {"id": "jev", "model": "jev-1.13.0", "version": "1.13.0"},
                "value": "tech",
                "probabilities": {"tech": 0.6, "energy": 0.4},
                "confidence": 0.6,
                "coverage": {"page_ranges": [[1, 1]], "est_tokens": 5},
            },
        },
    },
    "answers": {
        "outlook": {
            "kind": "score",
            "value": 2.0,
            "probabilities": {"low": 0.1, "high": 0.9},
            "confidence": 0.9,
            "classifier": {"id": "jev", "model": "jev-1.13.0", "version": "1.13.0"},
            "coverage": {"sections": ["mdna"], "est_tokens": 7},
        },
        "going_concern": {
            "value": None,
            "skipped": "missing section: controls",
            "coverage": {"est_tokens": 0},
        },
    },
    "usage": {"requests": 1, "input_tokens": 12, "output_tokens": 1},
    "timings": {"classify_ms": 5},
}


def test_the_full_schema_round_trips_through_json() -> None:
    record = DocumentRecord.model_validate(FULL)
    again = DocumentRecord.model_validate_json(record.model_dump_json())
    assert again == record
    assert record.metadata["cik"].source == "edgar-header"
    assert record.answers["going_concern"].value is None
    assert record.classification is not None
    assert record.classification.escalation.first is not None
    assert record.classification.escalation.first.value == "tech"


def test_a_minimal_record_still_validates() -> None:
    record = DocumentRecord.model_validate(
        {
            "record_id": "r",
            "job_id": "j",
            "status": "failed",
            "source": {"filename": "a.pdf", "format": "pdf", "sha256": "x", "bytes": 1},
        }
    )
    assert record.schema_version == SCHEMA_VERSION
    assert record.answers == {} and record.sections == [] and record.metadata == {}


def test_json_schema_is_generated_from_the_models() -> None:
    schema = record_json_schema()
    assert schema["title"] == "DocumentRecord"
    assert schema["properties"]["schema_version"]["default"] == SCHEMA_VERSION
    for field in ("external_id", "user_metadata", "metadata", "sections", "answers", "pages"):
        assert field in schema["properties"]
    assert schema["$defs"]["Page"]["properties"]["ocr_decision"]
