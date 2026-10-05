# SPDX-License-Identifier: Apache-2.0
"""The Workspace config file, `<workspace>/config.yaml` (docs/spec/v0.md §7.5, §11).

Two keys exist so far: `classifier.rate_limit` (the Classifier's rate ceiling, overriding the
default 80% of published limits) and `prices` (input USD per million tokens, by Classifier id).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from acceleread.ratelimit import MAX_IN_FLIGHT, RateLimit


class ConfigError(ValueError):
    """The config file is unreadable or holds an invalid value."""


@dataclass(frozen=True)
class Config:
    rate_limit: RateLimit | None = None
    prices: dict[str, float] = field(default_factory=dict)


def _positive(raw: Any, what: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float) or raw <= 0:
        raise ConfigError(f"{what} must be a positive number, got {raw!r}")
    return float(raw)


def _rate_limit(raw: Any) -> RateLimit:
    if not isinstance(raw, dict):
        raise ConfigError("classifier.rate_limit must be a mapping")
    unknown = set(raw) - {"tokens_per_s", "requests_per_s", "max_in_flight"}
    if unknown:
        raise ConfigError(f"classifier.rate_limit has unknown keys: {', '.join(sorted(unknown))}")
    try:
        return RateLimit(
            _positive(raw["tokens_per_s"], "classifier.rate_limit.tokens_per_s"),
            _positive(raw["requests_per_s"], "classifier.rate_limit.requests_per_s"),
            int(_positive(raw.get("max_in_flight", MAX_IN_FLIGHT), "rate_limit.max_in_flight")),
        )
    except KeyError as err:
        raise ConfigError(f"classifier.rate_limit needs {err.args[0]}") from err


def load_config(workspace: Path) -> Config:
    path = workspace / "config.yaml"
    if not path.is_file():
        return Config()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as err:
        raise ConfigError(f"{path}: {err}") from err
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    classifier = raw.get("classifier") or {}
    prices = raw.get("prices") or {}
    if not isinstance(classifier, dict) or not isinstance(prices, dict):
        raise ConfigError(f"{path}: `classifier` and `prices` must be mappings")
    return Config(
        rate_limit=_rate_limit(classifier["rate_limit"]) if "rate_limit" in classifier else None,
        prices={str(k): _positive(v, f"prices.{k}") for k, v in prices.items()},
    )
