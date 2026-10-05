# SPDX-License-Identifier: Apache-2.0
"""The versioned price table (docs/spec/v0.md §11). Costs are estimates, and cache hits cost $0."""

from collections.abc import Mapping

PRICE_TABLE_VERSION = "2026-10"

# USD per million input tokens, by Classifier id (the model name's first dash-separated word).
# Jev's output is free (spec §5.4). A model with no entry is unpriced: it costs $0 here, and the
# summary says so rather than guessing.
INPUT_USD_PER_MTOK: Mapping[str, float] = {"jev": 0.042}
OUTPUT_USD_PER_MTOK: Mapping[str, float] = {"jev": 0.0}


def classifier_family(model: str) -> str:
    return model.split("-", 1)[0]


def is_priced(model: str, overrides: Mapping[str, float] | None = None) -> bool:
    family = classifier_family(model)
    return family in INPUT_USD_PER_MTOK or family in (overrides or {})


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    overrides: Mapping[str, float] | None = None,
) -> float:
    """Estimated spend for tokens billed by `model`. `overrides` replaces input prices."""
    family = classifier_family(model)
    inputs = {**INPUT_USD_PER_MTOK, **(overrides or {})}
    return (
        input_tokens * inputs.get(family, 0.0)
        + output_tokens * OUTPUT_USD_PER_MTOK.get(family, 0.0)
    ) / 1_000_000
