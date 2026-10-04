# SPDX-License-Identifier: Apache-2.0
"""`jobs:validate`: every submit-time check, and the resolved manifest (spec §5.1, §7.1).

`validate()` never raises for a bad spec. It returns errors and warnings, so a surface can show
them all at once. `resolve()` is for callers that need the manifest and want a bad spec to stop.
"""

import difflib
import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from acceleread.classifier import Capabilities
from acceleread.jev import JEV_CAPABILITIES
from acceleread.models import (
    CANONICAL_SECTION_KEYS,
    OTHER,
    PRICE_TABLE_VERSION,
    ExtractionProfile,
    JobSettings,
    JobSpec,
    Question,
    QuestionSet,
    ResolvedManifest,
    ResolvedSet,
    Taxonomy,
    TaxonomyRef,
)

TAXONOMY_BUDGET_WARNING_SHARE = 0.20

# ISO 639-1 → Tesseract code (spec §4.3). A language outside this table is unknown.
TESSERACT_CODES = {
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "nl": "nld",
    "pl": "pol",
    "sv": "swe",
    "da": "dan",
    "no": "nor",
    "fi": "fin",
    "cs": "ces",
    "tr": "tur",
    "ru": "rus",
    "uk": "ukr",
    "el": "ell",
    "ar": "ara",
    "he": "heb",
    "hi": "hin",
    "ja": "jpn",
    "ko": "kor",
    "zh": "chi_sim",
}
# The core package vendors only `eng.traineddata` (spec §2). The Docker image adds more, and
# `acceleread ocr add-language` installs the rest into the Workspace; callers pass what they have.
VENDORED_OCR_LANGUAGES = frozenset({"en"})
# The `quality` Profile's RapidOCR model family is PP-OCR `latin`. It needs no Tesseract pack.
RAPIDOCR_LANGUAGES = frozenset(
    {"en", "de", "fr", "es", "it", "pt", "nl", "pl", "sv", "da", "no", "fi", "cs", "tr"}
)


@dataclass(frozen=True)
class Finding:
    code: str
    message: str


@dataclass(frozen=True)
class Estimate:
    """Cost and Classifier-bound duration. Stubbed until the price table and limiter exist."""

    documents: int
    cost_usd: float | None = None
    duration_seconds: float | None = None
    stubbed: bool = True


@dataclass
class ValidationReport:
    errors: list[Finding]
    warnings: list[Finding]
    estimate: Estimate
    manifest: ResolvedManifest | None = field(default=None)

    @property
    def ok(self) -> bool:
        return not self.errors


class SpecError(ValueError):
    def __init__(self, findings: list[Finding]) -> None:
        super().__init__("; ".join(f"{f.code}: {f.message}" for f in findings))
        self.findings = findings


def _hint(key: str, known: Iterable[str]) -> str:
    close = difflib.get_close_matches(key, list(known), n=1)
    return f" (did you mean '{close[0]}'?)" if close else ""


def _text_tokens(texts: Iterable[str | None], capabilities: Capabilities) -> float:
    return sum(len(t) for t in texts if t) / capabilities.chars_per_token


def _load_sets(paths: list[Path], errors: list[Finding]) -> list[tuple[Path, QuestionSet]]:
    loaded: list[tuple[Path, QuestionSet]] = []
    for path in paths:
        try:
            loaded.append((path, QuestionSet.from_file(path)))
        except (OSError, ValidationError, ValueError) as exc:  # includes YAML errors
            errors.append(Finding("question_set_unreadable", f"{path}: {exc}"))
    return loaded


def _check_options(
    label: str, count: int, capabilities: Capabilities, errors: list[Finding]
) -> None:
    if count > capabilities.max_choice_options:
        errors.append(
            Finding(
                "too_many_options",
                f"{label} has {count} options; the Classifier allows at most "
                f"{capabilities.max_choice_options}",
            )
        )


def _check_reads(
    label: str, reads: list[str] | None, known: Collection[str], errors: list[Finding]
) -> None:
    for key in reads or []:
        if key not in known:
            errors.append(
                Finding(
                    "unknown_reads_key",
                    f"{label} reads unknown Section key '{key}'{_hint(key, known)}",
                )
            )


def _check_ocr_language(
    language: str,
    profile: ExtractionProfile,
    installed: Collection[str],
    where: str,
    errors: list[Finding],
) -> None:
    def fail(message: str) -> None:
        errors.append(Finding("unsupported_ocr_language", f"{where}: {message}"))

    if language == "auto":
        fail("'auto' is reserved and not implemented in v0")
    elif language not in TESSERACT_CODES:
        fail(f"unknown language '{language}'{_hint(language, TESSERACT_CODES)}")
    elif profile == "quality":
        if language not in RAPIDOCR_LANGUAGES:
            fail(f"the quality Profile can't serve '{language}' (PP-OCR latin family only)")
    elif language not in installed:
        fail(
            f"no language pack installed for '{language}'; "
            f"run `acceleread ocr add-language {language}`"
        )


def _check(
    spec: JobSpec,
    capabilities: Capabilities,
    installed_languages: Collection[str],
    section_keys: Collection[str],
) -> tuple[list[Finding], list[Finding], ResolvedManifest | None]:
    errors: list[Finding] = []
    warnings: list[Finding] = []
    sets = _load_sets(spec.question_sets, errors)

    taxonomies: list[Taxonomy] = [spec.taxonomy] if spec.taxonomy else []
    taxonomies += [qs.taxonomy for _, qs in sets if qs.taxonomy]
    questions: list[Question] = [q for _, qs in sets for q in qs.questions] + spec.questions

    if not taxonomies and not questions and not errors:
        errors.append(Finding("no_judgments", "a Job needs at least one Taxonomy or Question"))
    if len(taxonomies) > 1:
        errors.append(
            Finding(
                "multiple_taxonomies", "a Job has at most one Taxonomy, but several were pulled in"
            )
        )

    names = [qs.name for _, qs in sets] + [t.name for t in taxonomies] + [q.name for q in questions]
    for name in sorted({n for n in names if names.count(n) > 1}):
        errors.append(Finding("duplicate_name", f"'{name}' is defined more than once"))

    budget_tokens = capabilities.token_budget * TAXONOMY_BUDGET_WARNING_SHARE

    def check_options_text(label: str, texts: list[str | None]) -> None:
        if _text_tokens(texts, capabilities) > budget_tokens:
            warnings.append(
                Finding(
                    "taxonomy_budget",
                    f"{label} uses over {TAXONOMY_BUDGET_WARNING_SHARE:.0%} of the "
                    f"{capabilities.token_budget}-token budget",
                )
            )

    for taxonomy in taxonomies:
        label = f"Taxonomy '{taxonomy.name}'"
        count = len(taxonomy.with_other().categories)
        _check_options(label, count, capabilities, errors)
        check_options_text(label, [t for c in taxonomy.categories for t in (c.name, c.description)])
        _check_reads(label, taxonomy.reads, section_keys, errors)
    for q in questions:
        label = f"Question '{q.name}'"
        if q.kind == "choice":
            _check_options(label, len(q.criteria), capabilities, errors)
        if q.kind == "choice":
            check_options_text(label, [t for c in q.criteria for t in (c.name, c.description)])
        _check_reads(label, q.reads, section_keys, errors)
        if not re.search(r"\bdocument\b", q.instructions, re.IGNORECASE):
            warnings.append(
                Finding(
                    "instructions_ignore_document",
                    f"Question '{q.name}' never mentions `document` in its instructions",
                )
            )

    inputs = {i.source for i in spec.inputs}
    for key in spec.overrides:
        if key not in inputs:
            errors.append(
                Finding("unknown_override", f"override for '{key}', which is not an input")
            )

    def check_languages(languages: list[str], profile: ExtractionProfile, where: str) -> None:
        for language in languages:
            _check_ocr_language(language, profile, installed_languages, where, errors)

    check_languages(spec.ocr_languages, spec.extraction_profile, "Job")
    for key, override in spec.overrides.items():
        if override.ocr_languages or override.extraction_profile:
            check_languages(
                override.ocr_languages or spec.ocr_languages,
                override.extraction_profile or spec.extraction_profile,
                f"Document '{key}'",
            )

    if errors:
        return errors, warnings, None
    return errors, warnings, _manifest(spec, sets, taxonomies)


def _manifest(
    spec: JobSpec, sets: list[tuple[Path, QuestionSet]], taxonomies: list[Taxonomy]
) -> ResolvedManifest:
    taxonomy = taxonomies[0].with_other() if taxonomies else None
    return ResolvedManifest(
        **{name: getattr(spec, name) for name in JobSettings.model_fields},
        taxonomy=taxonomy,
        taxonomy_ref=TaxonomyRef(name=taxonomy.name, hash=taxonomy.hash) if taxonomy else None,
        question_sets=[
            ResolvedSet(
                name=qs.name,
                version=qs.version,
                hash=qs.hash,
                source=str(path),
                taxonomy=qs.taxonomy,
                questions=qs.questions,
            )
            for path, qs in sets
        ],
        price_table_version=PRICE_TABLE_VERSION,
    )


def validate(
    spec: JobSpec,
    *,
    capabilities: Capabilities = JEV_CAPABILITIES,
    installed_languages: Collection[str] = VENDORED_OCR_LANGUAGES,
    extra_section_keys: Collection[str] = (),
) -> ValidationReport:
    """Dry-run every submit check. `extra_section_keys` are keys a detector declares."""
    keys = {*CANONICAL_SECTION_KEYS, *extra_section_keys} - {OTHER}
    errors, warnings, manifest = _check(spec, capabilities, installed_languages, keys)
    return ValidationReport(
        errors=errors,
        warnings=warnings,
        estimate=Estimate(documents=len(spec.inputs)),
        manifest=manifest,
    )


def resolve(
    spec: JobSpec,
    *,
    capabilities: Capabilities = JEV_CAPABILITIES,
    installed_languages: Collection[str] = VENDORED_OCR_LANGUAGES,
    extra_section_keys: Collection[str] = (),
) -> ResolvedManifest:
    """The resolved manifest, or `SpecError` carrying every validation error."""
    report = validate(
        spec,
        capabilities=capabilities,
        installed_languages=installed_languages,
        extra_section_keys=extra_section_keys,
    )
    if report.manifest is None:
        raise SpecError(report.errors)
    return report.manifest
