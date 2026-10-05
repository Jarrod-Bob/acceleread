# SPDX-License-Identifier: Apache-2.0
"""The price table and the Workspace config (spec §7.5, §11)."""

from pathlib import Path

import pytest

from acceleread.config import ConfigError, load_config
from acceleread.pricing import PRICE_TABLE_VERSION, estimate_cost_usd
from acceleread.ratelimit import RateLimit


def test_jev_costs_042_dollars_per_million_input_tokens_and_output_is_free() -> None:
    assert estimate_cost_usd("jev-1.13.0", 1_000_000, 5_000_000) == pytest.approx(0.042)


def test_an_unpriced_model_costs_nothing_rather_than_guessing() -> None:
    assert estimate_cost_usd("mystery-1", 1_000_000, 0) == 0.0


def test_the_price_table_is_versioned() -> None:
    assert PRICE_TABLE_VERSION != "unpriced-0"


def test_config_overrides_prices(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("prices:\n  jev: 1.0\n")
    config = load_config(tmp_path)
    assert estimate_cost_usd("jev-1.13.0", 2_000_000, 0, config.prices) == pytest.approx(2.0)


def test_classifier_rate_limit_key_sets_the_ceiling(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "classifier:\n  rate_limit:\n    tokens_per_s: 1000\n    requests_per_s: 2\n"
    )
    assert load_config(tmp_path).rate_limit == RateLimit(1000, 2)


def test_no_config_file_means_defaults(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    assert config.rate_limit is None and config.prices == {}


def test_a_bad_rate_limit_is_an_error_not_a_silent_default(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("classifier:\n  rate_limit:\n    tokens_per_s: -1\n")
    with pytest.raises(ConfigError, match="rate_limit"):
        load_config(tmp_path)
