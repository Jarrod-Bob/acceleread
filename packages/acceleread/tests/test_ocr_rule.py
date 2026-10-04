# SPDX-License-Identifier: Apache-2.0
"""The OCR rule (docs/spec/v0.md §4.2) as a pure function of per-Page signals."""

from acceleread.ocr_rule import PageSignals, Step3Outcome, decide

CLEAN = "Revenue grew eighteen percent on higher panel shipments. " * 3


def signals(text: str = CLEAN, image_coverage: float = 0.0, path_count: int = 0) -> PageSignals:
    return PageSignals(text=text, image_coverage=image_coverage, path_count=path_count)


def test_page_with_little_text_and_a_big_image_goes_to_ocr() -> None:
    verdict = decide(signals(text="  p. 3 ", image_coverage=0.6))
    assert verdict.ocr is True
    assert verdict.decision.step == 1
    assert verdict.decision.chars == 3
    assert verdict.decision.image_coverage == 0.6


def test_image_coverage_just_under_the_threshold_is_not_enough() -> None:
    assert decide(signals(text="", image_coverage=0.59)).ocr is False


def test_page_with_little_text_and_many_vector_paths_goes_to_ocr() -> None:
    verdict = decide(signals(text="", path_count=20))
    assert verdict.ocr is True
    assert verdict.decision.step == 1
    assert verdict.decision.path_count == 20


def test_blank_page_keeps_its_text_layer() -> None:
    verdict = decide(signals(text="", path_count=19, image_coverage=0.1))
    assert verdict.ocr is False
    assert verdict.decision.step == 1


def test_eighty_non_space_characters_is_enough_to_skip_step_one() -> None:
    eighty = "a" * 80
    assert decide(signals(text=eighty, image_coverage=0.9)).decision.step == 4
    assert decide(signals(text=eighty[:-1], image_coverage=0.9)).decision.step == 1


def test_spaces_do_not_count_towards_the_eighty_characters() -> None:
    text = ("a " * 79).strip()
    assert decide(signals(text=text, image_coverage=0.9)).ocr is True


def test_text_beside_a_large_image_is_kept() -> None:
    verdict = decide(signals(image_coverage=0.95))
    assert verdict.ocr is False
    assert verdict.decision.step == 4
    assert verdict.decision.image_coverage == 0.95


def test_five_percent_bad_characters_goes_to_ocr() -> None:
    text = "a" * 95 + "�" * 5
    verdict = decide(signals(text=text))
    assert verdict.ocr is True
    assert verdict.decision.step == 2
    assert verdict.decision.bad_char_ratio == 0.05


def test_each_kind_of_bad_character_counts() -> None:
    private_use, control, unassigned = "", "\x01", "͸"
    for bad in (private_use, control, unassigned):
        assert decide(signals(text="a" * 90 + bad * 10)).decision.step == 2


def test_newlines_and_tabs_are_not_bad_characters() -> None:
    verdict = decide(signals(text=("word\n\t" * 40)))
    assert verdict.decision.bad_char_ratio == 0.0
    assert verdict.ocr is False


def test_just_under_five_percent_bad_characters_is_kept() -> None:
    text = "a" * 96 + "�" * 4
    assert decide(signals(text=text)).ocr is False


def test_step_one_wins_over_step_two() -> None:
    verdict = decide(signals(text="�" * 10, image_coverage=0.9))
    assert verdict.decision.step == 1


def test_clean_text_layer_is_kept_at_step_four_without_a_step_three_hook() -> None:
    verdict = decide(signals())
    assert verdict.ocr is False
    assert verdict.decision.step == 4
    assert verdict.decision.word_ratio is None
    assert verdict.decision.jev_real_words is None


def test_step_three_hook_decides_when_provided() -> None:
    seen: list[str] = []

    def hook(text: str) -> Step3Outcome | None:
        seen.append(text)
        return Step3Outcome(ocr=True, word_ratio=0.4, jev_real_words=0.2, jev_skipped=False)

    verdict = decide(signals(), step3=hook)
    assert seen == [CLEAN]
    assert verdict.ocr is True
    assert verdict.decision.step == 3
    assert verdict.decision.word_ratio == 0.4
    assert verdict.decision.jev_real_words == 0.2
    assert verdict.decision.jev_skipped is False


def test_step_three_hook_may_decline_and_the_rule_falls_through() -> None:
    verdict = decide(signals(), step3=lambda text: None)
    assert verdict.decision.step == 4
    assert verdict.ocr is False


def test_step_three_hook_keeping_the_page_reports_step_three() -> None:
    outcome = Step3Outcome(ocr=False, word_ratio=0.65, jev_real_words=0.9, jev_skipped=False)
    verdict = decide(signals(), step3=lambda text: outcome)
    assert verdict.ocr is False
    assert verdict.decision.step == 3


def test_steps_one_and_two_never_call_the_hook() -> None:
    def hook(text: str) -> Step3Outcome | None:
        raise AssertionError("step 3 must not run")

    decide(signals(text="", path_count=30), step3=hook)
    decide(signals(text="a" * 90 + "�" * 10), step3=hook)
