# SPDX-License-Identifier: Apache-2.0
"""`jobs:validate`: every submit-time check (docs/spec/v0.md §5.1, §4.3, §7.1)."""

from pathlib import Path

import pytest

from acceleread import JobSpec
from acceleread.classifier import Capabilities
from acceleread.models import (
    PRICE_TABLE_VERSION,
    Category,
    DocumentOverride,
    Question,
    QuestionSet,
    Taxonomy,
)
from acceleread.validate import Finding, SpecError, ValidationReport, resolve, validate

SETS = Path(__file__).parent / "fixtures" / "sets"
FILING_RISK = SETS / "filing_risk.yaml"


def taxonomy(*names: str, description: str | None = None) -> Taxonomy:
    return Taxonomy(name="t", categories=[Category(name=n, description=description) for n in names])


def noul(name: str = "q", instructions: str = "Is `document` a filing?", **kw: object) -> Question:
    return Question(name=name, kind="noul", instructions=instructions, **kw)  # type: ignore[arg-type]


def spec(**kw: object) -> JobSpec:
    kw.setdefault("inputs", ["a.pdf"])
    return JobSpec.model_validate(kw)


def codes(findings: list[Finding]) -> list[str]:
    return [f.code for f in findings]


def test_a_taxonomy_alone_or_a_question_alone_is_a_valid_job() -> None:
    assert validate(spec(taxonomy=taxonomy("a"))).ok
    assert validate(spec(questions=[noul()])).ok


def test_a_job_needs_a_taxonomy_or_a_question() -> None:
    report = validate(spec())
    assert not report.ok and codes(report.errors) == ["no_judgments"]


def test_a_taxonomy_from_a_question_set_counts() -> None:
    assert validate(spec(question_sets=[FILING_RISK])).ok


def test_names_must_be_unique_across_everything_pulled_in() -> None:
    clash = noul("going_concern")  # also defined by the Set
    report = validate(spec(question_sets=[FILING_RISK], questions=[clash]))
    assert codes(report.errors) == ["duplicate_name"]
    assert "going_concern" in report.errors[0].message

    twice = validate(spec(questions=[noul("x"), noul("x")]))
    assert codes(twice.errors) == ["duplicate_name"]

    sets_clash = validate(spec(question_sets=[FILING_RISK, FILING_RISK]))
    assert "duplicate_name" in codes(sets_clash.errors)


def test_two_taxonomies_in_one_job_is_an_error() -> None:
    report = validate(spec(taxonomy=taxonomy("a"), question_sets=[FILING_RISK]))
    assert codes(report.errors) == ["multiple_taxonomies"]


def test_an_unreadable_question_set_is_reported_not_raised(tmp_path: Path) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text("name: x\nversion: 1\nquestions: [{name: q, kind: nope}]\n")
    report = validate(spec(question_sets=[broken, tmp_path / "missing.yaml"], questions=[noul()]))
    assert codes(report.errors) == ["question_set_unreadable"] * 2


def test_option_limit_counts_the_automatic_other() -> None:
    names = [f"c{i}" for i in range(255)]
    assert codes(validate(spec(taxonomy=taxonomy(*names))).errors) == ["too_many_options"]
    assert validate(spec(taxonomy=taxonomy(*names[:254]))).ok  # 254 + automatic other = 255
    assert validate(spec(taxonomy=taxonomy(*names[:254], "other"))).ok  # user-defined other


def test_option_limit_follows_the_classifier_capabilities() -> None:
    small = Capabilities(
        kinds=frozenset({"choice"}), max_choice_options=4, token_budget=32_000, chars_per_token=3.0
    )
    assert codes(
        validate(spec(taxonomy=taxonomy("a", "b", "c", "d")), capabilities=small).errors
    ) == ["too_many_options"]


def test_a_choice_question_has_the_limit_but_no_automatic_other() -> None:
    options = [Category(name=f"o{i}") for i in range(255)]
    ok = Question(name="c", kind="choice", instructions="Pick for `document`", criteria=options)
    assert validate(spec(questions=[ok])).ok
    over = ok.model_copy(update={"criteria": [*options, Category(name="o255")]})
    assert codes(validate(spec(questions=[over])).errors) == ["too_many_options"]


def test_a_taxonomy_over_a_fifth_of_the_budget_warns() -> None:
    wordy = taxonomy("a", "b", description="x" * 20_000)  # ~13k tokens at 3 chars/token, > 20%
    report = validate(spec(taxonomy=wordy))
    assert report.ok and codes(report.warnings) == ["taxonomy_budget"]
    assert codes(validate(spec(taxonomy=taxonomy("a", "b", description="x" * 100))).warnings) == []


def test_instructions_that_never_mention_document_warn() -> None:
    report = validate(spec(questions=[noul("a", "Is this a filing?"), noul("b")]))
    assert report.ok
    assert codes(report.warnings) == ["instructions_ignore_document"]
    assert "'a'" in report.warnings[0].message


def test_unknown_reads_keys_suggest_the_nearest_canonical_key() -> None:
    q = noul("q", reads=["risk_factor", "mdna", "zzz"])
    report = validate(spec(questions=[q]))
    assert codes(report.errors) == ["unknown_reads_key"] * 2
    assert "did you mean 'risk_factors'" in report.errors[0].message
    assert "did you mean" not in report.errors[1].message
    # a Taxonomy's reads are checked too; detector-declared keys can be allowed
    tax = taxonomy("a").model_copy(update={"reads": ["bussiness"]})
    assert "did you mean 'business'" in validate(spec(taxonomy=tax)).errors[0].message
    extra = validate(spec(questions=[noul(reads=["custom"])]), extra_section_keys={"custom"})
    assert extra.ok


def test_other_cannot_be_read() -> None:
    assert codes(validate(spec(questions=[noul(reads=["other"])])).errors) == ["unknown_reads_key"]


def test_ocr_languages_must_be_known_and_installed() -> None:
    assert validate(spec(taxonomy=taxonomy("a"), ocr_languages=["en"])).ok
    de = validate(spec(taxonomy=taxonomy("a"), ocr_languages=["de"]))  # only English is vendored
    assert codes(de.errors) == ["unsupported_ocr_language"]
    assert "ocr add-language de" in de.errors[0].message
    assert validate(
        spec(taxonomy=taxonomy("a"), ocr_languages=["de"]), installed_languages={"en", "de"}
    ).ok
    unknown = validate(spec(taxonomy=taxonomy("a"), ocr_languages=["xx"]))
    assert codes(unknown.errors) == ["unsupported_ocr_language"]
    auto = validate(spec(taxonomy=taxonomy("a"), ocr_languages=["auto"]))
    assert "not implemented" in auto.errors[0].message
    missing = validate(spec(taxonomy=taxonomy("a"), ocr_languages=["ja"]))
    assert "ocr add-language ja" in missing.errors[0].message
    installed = validate(
        spec(taxonomy=taxonomy("a"), ocr_languages=["ja"]), installed_languages={"en", "ja"}
    )
    assert installed.ok


def test_the_quality_profile_checks_rapidocr_support_not_tesseract_packs() -> None:
    # RapidOCR's latin family serves German without any Tesseract pack installed.
    latin = spec(taxonomy=taxonomy("a"), extraction_profile="quality", ocr_languages=["de"])
    assert validate(latin).ok
    # Japanese is outside the latin family, even with its Tesseract pack installed.
    ja = spec(taxonomy=taxonomy("a"), extraction_profile="quality", ocr_languages=["ja"])
    report = validate(ja, installed_languages={"en", "ja"})
    assert codes(report.errors) == ["unsupported_ocr_language"]
    assert "quality" in report.errors[0].message
    # the fast Profile still needs the pack
    fast = spec(taxonomy=taxonomy("a"), ocr_languages=["ja"])
    assert validate(fast, installed_languages={"en", "ja"}).ok


def test_per_document_overrides_are_validated_with_their_own_profile() -> None:
    job = spec(
        inputs=["a.pdf", {"source": "b.pdf"}],
        taxonomy=taxonomy("a"),
        overrides={"b.pdf": DocumentOverride(ocr_languages=["xx"])},
    )
    assert codes(validate(job).errors) == ["unsupported_ocr_language"]
    stray = spec(taxonomy=taxonomy("a"), overrides={"nope.pdf": DocumentOverride()})
    assert codes(validate(stray).errors) == ["unknown_override"]


def test_there_is_no_estimate_when_no_input_can_be_read() -> None:
    estimate = validate(spec(taxonomy=taxonomy("a"))).estimate
    assert estimate.documents == 1
    assert estimate.cost_usd is None and estimate.duration_seconds is None
    assert estimate.stubbed


def test_report_helpers() -> None:
    report = validate(spec(taxonomy=taxonomy("a")))
    assert isinstance(report, ValidationReport) and report.manifest is not None
    assert validate(spec()).manifest is None  # nothing to resolve when the spec is invalid


def test_manifest_inlines_sets_with_name_version_and_hash() -> None:
    job = spec(question_sets=[FILING_RISK], questions=[noul("extra")], max_cost_usd=5)
    manifest = resolve(job)
    (resolved_set,) = manifest.question_sets
    loaded = QuestionSet.from_file(FILING_RISK)
    assert (resolved_set.name, resolved_set.version) == ("filing-risk", "2")
    assert resolved_set.hash == loaded.hash and resolved_set.questions == loaded.questions
    assert [q.name for q in manifest.questions] == ["extra"]
    assert manifest.price_table_version == PRICE_TABLE_VERSION
    assert manifest.max_cost_usd == 5


def test_manifest_fills_defaults_and_adds_other_to_the_taxonomy() -> None:
    manifest = resolve(spec(taxonomy=taxonomy("a", "b")))
    assert manifest.extraction_profile == "fast" and manifest.ocr_languages == ["en"]
    assert manifest.cache is True and manifest.llm_escalation.escalation_max == 0.02
    assert manifest.taxonomy is not None
    assert [c.name for c in manifest.taxonomy.categories] == ["a", "b", "other"]
    assert manifest.taxonomy_ref is not None
    assert manifest.taxonomy_ref.hash == manifest.taxonomy.hash


def test_manifest_takes_the_taxonomy_from_a_question_set() -> None:
    manifest = resolve(spec(question_sets=[FILING_RISK]))
    assert manifest.taxonomy is not None and manifest.taxonomy.name == "filing-type"
    assert manifest.taxonomy.categories[-1].name == "other"


def test_resolving_an_invalid_spec_raises_with_the_findings() -> None:
    with pytest.raises(SpecError) as caught:
        resolve(spec())
    assert codes(caught.value.findings) == ["no_judgments"]


def test_an_unreadable_set_does_not_also_report_no_judgments(tmp_path: Path) -> None:
    report = validate(spec(question_sets=[tmp_path / "missing.yaml"]))
    assert codes(report.errors) == ["question_set_unreadable"]


def test_the_taxonomy_name_joins_the_namespace_and_each_clash_is_reported_once() -> None:
    clash = validate(spec(taxonomy=taxonomy("a"), questions=[noul("t")]))  # Taxonomy is named "t"
    assert codes(clash.errors) == ["duplicate_name"]
    thrice = validate(spec(questions=[noul("x"), noul("x"), noul("x")]))
    assert codes(thrice.errors) == ["duplicate_name"]


def test_a_score_question_is_not_subject_to_the_budget_warning() -> None:
    wordy = [Category(name="lo", description="x" * 30_000), Category(name="hi")]
    score = Question(name="s", kind="score", instructions="Rate `document`", criteria=wordy)
    assert validate(spec(questions=[score])).warnings == []
    choice = score.model_copy(update={"kind": "choice"})
    assert codes(validate(spec(questions=[choice])).warnings) == ["taxonomy_budget"]


def test_the_manifest_carries_every_job_spec_field() -> None:
    from acceleread.models import ResolvedManifest

    assert set(JobSpec.model_fields) - {"question_sets"} <= set(ResolvedManifest.model_fields)
