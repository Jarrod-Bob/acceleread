# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for Jobs, Taxonomies and Document Records (docs/spec/v0.md §5.1, §6, §7.1).

The tracer carries only the fields its thin slice fills; later build issues add the rest.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1.0"
DEFAULT_JEV_MODEL = "jev-1.13.0"
OTHER = "other"


class Category(BaseModel):
    name: str
    description: str | None = None


class Taxonomy(BaseModel):
    """A flat set of Categories. `other` is added unless the user defines one."""

    name: str
    categories: list[Category] = Field(min_length=1)

    @classmethod
    def from_file(cls, path: Path) -> "Taxonomy":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def with_other(self) -> "Taxonomy":
        if any(c.name == OTHER for c in self.categories):
            return self
        other = Category(name=OTHER, description="None of the other categories fits.")
        return self.model_copy(update={"categories": [*self.categories, other]})

    @property
    def hash(self) -> str:
        canonical = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


class JobSpec(BaseModel):
    inputs: list[Path] = Field(min_length=1)
    taxonomy: Taxonomy
    model: str = DEFAULT_JEV_MODEL


class TaxonomyRef(BaseModel):
    name: str
    hash: str


class ClassifierInfo(BaseModel):
    id: str
    model: str
    version: str


class Coverage(BaseModel):
    """The part of a Document a single Judgment read."""

    page_ranges: list[tuple[int, int]] = Field(default_factory=list)
    truncated: bool = False
    est_tokens: int
    input_tokens: int | None = None


class Escalation(BaseModel):
    status: Literal["none", "flagged", "escalated", "failed"] = "none"


class Judgment(BaseModel):
    kind: Literal["noul", "score", "choice"]
    value: str | float
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    classifier: ClassifierInfo
    coverage: Coverage
    escalation: Escalation = Field(default_factory=Escalation)


class Page(BaseModel):
    number: int
    start: int
    end: int
    method: Literal["text-layer", "ocr-full", "ocr-regions"] = "text-layer"


class Source(BaseModel):
    filename: str
    format: Literal["pdf", "html"]
    sha256: str
    bytes: int


class RecordError(BaseModel):
    stage: Literal["extract", "classify"]
    code: str
    message: str
    page: int | None = None


class Usage(BaseModel):
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    ocr_pages: int = 0
    cache_hits: int = 0


class Timings(BaseModel):
    queued_ms: int = 0
    extract_ms: int = 0
    ocr_ms: int = 0
    classify_ms: int = 0
    escalate_ms: int = 0


class DocumentRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    record_id: str
    job_id: str
    status: Literal["ok", "partial", "failed", "cancelled"]
    errors: list[RecordError] = Field(default_factory=list)
    attempts: int = 1
    source: Source
    extraction_profile: Literal["fast", "quality"] = "fast"
    text: str | None = None
    pages: list[Page] = Field(default_factory=list)
    taxonomy: TaxonomyRef | None = None
    classification: Judgment | None = None
    usage: Usage = Field(default_factory=Usage)
    timings: Timings = Field(default_factory=Timings)
