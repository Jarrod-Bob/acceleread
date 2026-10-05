# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for Jobs, Taxonomies and Document Records (docs/spec/v0.md §5.1, §6, §7.1).

The tracer carries only the fields its thin slice fills; later build issues add the rest.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "0.1.0"
DEFAULT_JEV_MODEL = "jev-1.13.0"
DEFAULT_ESCALATION_MODEL = "claude-opus-5-5"
DEFAULT_ESCALATION_MAX = 0.02  # share of a Job's Documents that may be escalated
PRICE_TABLE_VERSION = "unpriced-0"  # placeholder until the versioned price table exists
OTHER = "other"

type ExtractionProfile = Literal["fast", "quality"]
# The closed set of canonical Section keys (spec §4.4). Everything else is `other`, unreadable.
CANONICAL_SECTION_KEYS = (
    "business",
    "risk_factors",
    "cybersecurity",
    "legal_proceedings",
    "mdna",
    "market_risk",
    "financial_statements",
    "controls",
    "governance",
    "exhibits",
)


class _Strict(BaseModel):
    """Base for everything a user writes in a spec file: a misspelt field is an error."""

    model_config = ConfigDict(extra="forbid")


class Category(_Strict):
    name: str
    description: str | None = None


def _canonical_hash(model: BaseModel) -> str:
    """Unset optional fields are left out, so adding one later doesn't change old hashes."""
    dumped = model.model_dump(mode="json", exclude_none=True)
    canonical = json.dumps(dumped, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _reject_duplicate_names(what: str, items: list[Category]) -> None:
    names = [c.name for c in items]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate {what} names: {', '.join(dupes)}")


class Taxonomy(_Strict):
    """A flat set of Categories. `other` is added unless the user defines one."""

    name: str
    categories: list[Category] = Field(min_length=1)
    reads: list[str] | None = None  # canonical Section keys; None reads the whole Document
    escalate_below: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _unique_categories(self) -> Self:
        _reject_duplicate_names("Category", self.categories)
        return self

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
        return _canonical_hash(self)


class Question(_Strict):
    """A user-defined Judgment: a Noul, a Score (ordered criteria) or a Choice (named options)."""

    name: str
    kind: Literal["noul", "score", "choice"]
    instructions: str
    criteria: list[Category] = Field(default_factory=list)  # Score: low to high; Choice: options
    reads: list[str] | None = None  # canonical Section keys; None reads the whole Document
    fallback: Literal["head_tail"] | None = None
    escalate_below: float | None = Field(default=None, ge=0, le=1)

    @field_validator("criteria", mode="before")
    @classmethod
    def _bare_names(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [{"name": v} if isinstance(v, str) else v for v in value]
        return value

    @model_validator(mode="after")
    def _criteria_fit_kind(self) -> Self:
        if self.kind == "noul" and self.criteria:
            raise ValueError("a noul Question takes no criteria")
        if self.kind != "noul" and len(self.criteria) < 2:
            raise ValueError(f"a {self.kind} Question needs at least two criteria")
        _reject_duplicate_names("criteria", self.criteria)
        return self


class QuestionSet(_Strict):
    """A named, versioned file of Questions plus an optional Taxonomy."""

    name: str
    version: str
    taxonomy: Taxonomy | None = None
    questions: list[Question] = Field(default_factory=list)

    @field_validator("version", mode="before")
    @classmethod
    def _version_as_text(cls, value: Any) -> Any:
        return (
            str(value) if isinstance(value, int | float) and not isinstance(value, bool) else value
        )

    @classmethod
    def from_file(cls, path: Path) -> "QuestionSet":
        """Load YAML or JSON (JSON is valid YAML)."""
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    @property
    def hash(self) -> str:
        return _canonical_hash(self)


class LLMEscalation(_Strict):
    """Opt-in automatic Escalation of low-confidence Judgments to an LLM (spec §5.5)."""

    enabled: bool = False
    model: str = DEFAULT_ESCALATION_MODEL
    escalation_max: float = Field(default=DEFAULT_ESCALATION_MAX, ge=0, le=1)


class DocumentOverride(_Strict):
    """Per-Document overrides of the Job's Extraction Profile and OCR languages."""

    extraction_profile: ExtractionProfile | None = None
    ocr_languages: list[str] | None = Field(default=None, min_length=1)


class DocumentInput(_Strict):
    """One input: a local path, glob or URL, kept verbatim, with optional caller-supplied data."""

    source: str
    external_id: str | None = None
    user_metadata: dict[str, Any] = Field(default_factory=dict)


class JobSettings(_Strict):
    """The Job fields a manifest carries over unchanged from the spec."""

    inputs: list[DocumentInput] = Field(min_length=1)
    questions: list[Question] = Field(default_factory=list)
    extraction_profile: ExtractionProfile = "fast"
    ocr_languages: list[str] = Field(default_factory=lambda: ["en"], min_length=1)
    overrides: dict[str, DocumentOverride] = Field(default_factory=dict)  # keyed by input path
    escalate_below: float | None = Field(default=None, ge=0, le=1)
    llm_escalation: LLMEscalation = Field(default_factory=LLMEscalation)
    max_cost_usd: float | None = Field(default=None, gt=0)
    cache: bool = True
    user_agent: str | None = None
    model: str = DEFAULT_JEV_MODEL

    @field_validator("inputs", mode="before")
    @classmethod
    def _plain_sources(cls, value: Any) -> Any:
        """A bare string or path is a source; it is never resolved or turned into a Path."""
        if not isinstance(value, list):
            return value
        return [{"source": os.fspath(v)} if isinstance(v, str | os.PathLike) else v for v in value]


class JobSpec(JobSettings):
    """The one Job spec used by the library, the HTTP API and the CLI (spec §7.1)."""

    taxonomy: Taxonomy | None = None
    question_sets: list[Path] = Field(default_factory=list)


class TaxonomyRef(BaseModel):
    name: str
    hash: str


class ResolvedSet(BaseModel):
    """A Question Set inlined into a manifest, pinned by name, version and content hash."""

    name: str
    version: str
    hash: str
    source: str  # where it was loaded from
    taxonomy: Taxonomy | None = None
    questions: list[Question]


class ResolvedManifest(JobSettings):
    """The resolved Job spec: Sets inlined, defaults filled, price-table version recorded.

    Immutable once the Job starts, except `max_cost_usd` (spec §7.1).
    """

    taxonomy: Taxonomy | None  # the Job's or its Set's, with `other` added
    taxonomy_ref: TaxonomyRef | None
    question_sets: list[ResolvedSet]
    price_table_version: str


class ClassifierInfo(BaseModel):
    id: str
    model: str
    version: str


class Span(BaseModel):
    start: int
    end: int

    @model_validator(mode="after")
    def _ordered(self) -> "Span":
        if self.end < self.start:
            raise ValueError("a span cannot end before it starts")
        return self


class Coverage(BaseModel):
    """The part of a Document a single Judgment read."""

    sections: list[str] = Field(default_factory=list)
    spans: list[Span] = Field(default_factory=list)  # exact character offsets read
    page_ranges: list[tuple[int, int]] = Field(default_factory=list)  # Pages those spans touch
    truncated: bool = False
    shrunk: bool = False
    est_tokens: int
    input_tokens: int | None = None
    note: str | None = None


class FirstPass(BaseModel):
    """The first-pass result an Escalation replaced."""

    classifier: ClassifierInfo
    value: str | float
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    coverage: Coverage


class Escalation(BaseModel):
    status: Literal["none", "flagged", "escalated", "failed"] = "none"
    reason: str | None = None
    first: FirstPass | None = None


class Judgment(BaseModel):
    kind: Literal["noul", "score", "choice"]
    value: str | float
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    classifier: ClassifierInfo
    coverage: Coverage
    escalation: Escalation = Field(default_factory=Escalation)


class SkippedAnswer(BaseModel):
    """An Answer skipped because a Section the Question reads is missing."""

    value: None = None
    skipped: str
    coverage: Coverage


class OcrDecision(BaseModel):
    """Why the OCR rule sent a Page to OCR or kept its text layer (spec §4.2, §6)."""

    step: int
    reason: str
    chars: int
    word_ratio: float | None = None
    bad_char_ratio: float
    image_coverage: float
    path_count: int
    jev_real_words: float | None = None
    jev_skipped: bool = False


class Page(BaseModel):
    number: int
    start: int
    end: int
    method: Literal["text-layer", "ocr-full", "ocr-regions"] = "text-layer"
    engine: str | None = None
    engine_version: str | None = None
    ocr_languages: list[str] = Field(default_factory=list)
    ocr_confidence: float | None = None
    ocr_decision: OcrDecision | None = None
    image_coverage: float | None = None


class Verification(BaseModel):
    status: str
    p: float | None = None


class Section(BaseModel):
    keys: list[str]
    label: str
    form_ref: str | None = None
    spans: list[Span]
    method: str
    confidence: float | None = None
    verification: Verification | None = None
    flags: list[str] = Field(default_factory=list)
    est_tokens: int = 0


class MetadataField(BaseModel):
    """A metadata value and where it came from. Metadata comes from code only."""

    value: str | int | float | bool | None
    source: str


class Source(BaseModel):
    filename: str
    format: Literal["pdf", "html"]
    sha256: str
    bytes: int
    group: str | None = None  # e.g. the accession number joining an exhibit to its filing


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
    external_id: str | None = None
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "partial", "failed", "cancelled"]
    errors: list[RecordError] = Field(default_factory=list)
    attempts: int = 1
    source: Source
    metadata: dict[str, MetadataField] = Field(default_factory=dict)
    extraction_profile: ExtractionProfile = "fast"
    text: str | None = None
    pages: list[Page] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    taxonomy: TaxonomyRef | None = None
    classification: Judgment | None = None
    answers: dict[str, Judgment | SkippedAnswer] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    timings: Timings = Field(default_factory=Timings)


def record_json_schema() -> dict[str, Any]:
    """The Document Record's JSON Schema, generated from the Pydantic models."""
    return DocumentRecord.model_json_schema()
