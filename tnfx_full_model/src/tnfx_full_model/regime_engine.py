from __future__ import annotations

from itertools import product
from typing import Mapping

from .constants import REGIME_FLAGS
from .family_specs import inverse_cdf


def state_id(state: Mapping[str, int]) -> str:
    return "".join(str(int(state[f])) for f in REGIME_FLAGS)


def is_feasible_regime(family: str, state: Mapping[str, int]) -> bool:
    if state["g_TE"] > state["g_PAE"]:
        return False
    if family == "exporter" and state["g_CIRC"] != 0:
        return False
    if family in {"importer", "processor", "trader"} and state["g_CIRC"] not in {0, 1}:
        return False
    if family == "importer" and state["g_PAE"] == 1:
        return False
    if state["g_NR"] == 1 and state["g_PAE"] != 1:
        return False
    if state["g_PAE"] == 1 and state["g_ACC"] != 1:
        return False
    return True


def enumerate_feasible_regimes(family: str, model_config: Mapping) -> list[dict]:
    weights = model_config["regime_score_weights"]
    states = []
    for bits in product([0, 1], repeat=len(REGIME_FLAGS)):
        raw = dict(zip(REGIME_FLAGS, bits))
        if not is_feasible_regime(family, raw):
            continue
        score = float(weights.get("base", 1.0))
        for flag, value in raw.items():
            if value:
                score *= float(weights.get(flag, 1.0))
        row = {"regime_state": state_id(raw), **raw, "score": score}
        states.append(row)
    total = sum(s["score"] for s in states)
    for row in states:
        row["probability"] = row["score"] / total
    return states


def feasible_regime_priors(family: str, model_config: Mapping) -> dict[str, float]:
    return {row["regime_state"]: row["probability"] for row in enumerate_feasible_regimes(family, model_config)}


def all_regime_priors(model_config: Mapping):
    rows = []
    for family in model_config["families"]:
        for row in enumerate_feasible_regimes(family, model_config):
            out = {"family": family, "prior_label": "author_calibrated", **row}
            rows.append(out)
    return rows


def select_regime(u: float, family: str, model_config: Mapping) -> dict:
    regimes = enumerate_feasible_regimes(family, model_config)
    selected = inverse_cdf(u, {r["regime_state"]: r["probability"] for r in regimes})
    for row in regimes:
        if row["regime_state"] == selected:
            return row.copy()
    raise RuntimeError("Selected regime not found")


def apply_regime_effects(values: dict, family_spec: Mapping, regime: Mapping, model_config: Mapping) -> dict:
    adjusted = values.copy()
    effects = model_config.get("regime_bound_effects", {})
    for flag in REGIME_FLAGS:
        if int(regime.get(flag, 0)) != 1:
            continue
        for var, rule in effects.get(flag, {}).items():
            if var not in adjusted:
                continue
            base_bounds = family_spec[var]
            width = float(base_bounds[1]) - float(base_bounds[0])
            adjusted[var] = adjusted[var] + float(rule.get("shift_pct", 0.0)) * width
            adjusted[var] = min(max(adjusted[var], float(base_bounds[0])), float(base_bounds[1]))
    clip = model_config.get("lambda_clip", {"lower": 0.0, "upper": 0.95})
    adjusted["lambda"] = min(max(adjusted["lambda"], clip["lower"]), clip["upper"])
    return adjusted
