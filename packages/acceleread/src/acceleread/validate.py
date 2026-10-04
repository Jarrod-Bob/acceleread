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
# The Docker image bundles these `tessdata_fast` packs (spec §2).
BUNDLED_OCR_LANGUAGES = frozenset({"en", "de", "fr", "es", "it", "pt", "nl"})
# The `quality` Profile's RapidOCR model family is PP-OCR `latin`.
QUALITY_OCR_LANGUAGES = frozenset(
    {"en", "de", "fr", "es", "it", "pt", "nl", "pl", "sv", "da", "no", "fi", "cs", "tr"}
)


@dataclass(frozen=True)
class Issue:
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
    errors: list[Issue]
    warnings: list[Issue]
    estimate: Estimate
    manifest: ResolvedManifest | None = field(default=None)

    @property
    def ok(self) -> bool:
        return not self.errors


class SpecError(ValueError):
    def __init__(self, issues: list[Issue]) -> None:
        super().__init__("; ".join(f"{i.code}: {i.message}" for i in issues))
        self.issues = issues


def _hint(key: str, known: Iterable[str]) -> str:
    close = difflib.get_close_matches(key, list(known), n=1)
    return f" (did you mean '{close[0]}'?)" if close else ""


def _text_tokens(texts: Iterable[str | None], capabilities: Capabilities) -> float:
    return sum(len(t) for t in texts if t) / capabilities.chars_per_token


def _load_sets(paths: list[Path], errors: list[Issue]) -> list[tuple[Path, QuestionSet]]:
    loaded: list[tuple[Path, QuestionSet]] = []
    for path in paths:
        try:
            loaded.append((path, QuestionSet.from_file(path)))
        except (OSError, ValidationError, ValueError) as exc:  # includes YAML errors
            errors.append(Issue("question_set_unreadable", f"{path}: {exc}"))
    return loaded


def _check_options(label: str, count: int, capabilities: Capabilities, errors: list[Issue]) -> None:
    if count > capabilities.max_choice_options:
        errors.append(
            Issue(
                "too_many_options",
                f"{label} has {count} options; the Classifier allows at most "
                f"{capabilities.max_choice_options}",
            )
        )


def _check_reads(
    label: str, reads: list[str] | None, known: Collection[str], errors: list[Issue]
) -> None:
    for key in reads or []:
        if key not in known:
            errors.append(
                Issue(
                    "unknown_reads_key",
                    f"{label} reads unknown Section key '{key}'{_hint(key, known)}",
                )
            )


def _check_ocr_language(
    language: str,
    profile: str,
    installed: Collection[str],
    where: str,
    errors: list[Issue],
) -> None:
    def fail(message: str) -> None:
        errors.append(Issue("unsupported_ocr_language", f"{where}: {message}"))

    if language == "auto":
        fail("'auto' is reserved and not implemented in v0")
    elif language not in TESSERACT_CODES:
        fail(f"unknown language '{language}'{_hint(language, TESSERACT_CODES)}")
    elif profile == "quality" and language not in QUALITY_OCR_LANGUAGES:
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
) -> tuple[list[Issue], list[Issue], ResolvedManifest | None]:
    errors: list[Issue] = []
    warnings: list[Issue] = []
    sets = _load_sets(spec.question_sets, errors)

    taxonomies: list[Taxonomy] = [spec.taxonomy] if spec.taxonomy else []
    taxonomies += [qs.taxonomy for _, qs in sets if qs.taxonomy]
    questions: list[Question] = [q for _, qs in sets for q in qs.questions] + spec.questions

    if not taxonomies and not questions and not errors:
        errors.append(Issue("no_judgments", "a Job needs at least one Taxonomy or Question"))
    if len(taxonomies) > 1:
        errors.append(
            Issue(
                "multiple_taxonomies", "a Job has at most one Taxonomy, but several were pulled in"
            )
        )

    seen: set[str] = set()
    names = [qs.name for _, qs in sets] + [q.name for q in questions]
    for name in names:
        if name in seen:
            errors.append(Issue("duplicate_name", f"'{name}' is defined more than once"))
        seen.add(name)

    budget_chars = capabilities.token_budget * TAXONOMY_BUDGET_WARNING_SHARE

    def check_options_text(label: str, texts: list[str | None]) -> None:
        if _text_tokens(texts, capabilities) > budget_chars:
            warnings.append(
                Issue(
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
        if q.kind != "noul":
            check_options_text(label, [t for c in q.criteria for t in (c.name, c.description)])
        _check_reads(label, q.reads, section_keys, errors)
        if not re.search(r"\bdocument\b", q.instructions, re.IGNORECASE):
            warnings.append(
                Issue(
                    "instructions_ignore_document",
                    f"Question '{q.name}' never mentions `document` in its instructions",
                )
            )

    inputs = {str(p) for p in spec.inputs}
    for key in spec.overrides:
        if key not in inputs:
            errors.append(Issue("unknown_override", f"override for '{key}', which is not an input"))

    def check_languages(languages: list[str], profile: str, where: str) -> None:
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
        inputs=spec.inputs,
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
        questions=spec.questions,
        extraction_profile=spec.extraction_profile,
        ocr_languages=spec.ocr_languages,
        overrides=spec.overrides,
        escalate_below=spec.escalate_below,
        llm_escalation=spec.llm_escalation,
        max_cost_usd=spec.max_cost_usd,
        cache=spec.cache,
        user_agent=spec.user_agent,
        model=spec.model,
        price_table_version=PRICE_TABLE_VERSION,
    )


def validate(
    spec: JobSpec,
    *,
    capabilities: Capabilities = JEV_CAPABILITIES,
    installed_languages: Collection[str] = BUNDLED_OCR_LANGUAGES,
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
    installed_languages: Collection[str] = BUNDLED_OCR_LANGUAGES,
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
