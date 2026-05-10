from __future__ import annotations

from typing import Mapping

from .constants import FAMILIES


def load_families(model_config: Mapping) -> dict:
    families = dict(model_config["families"])
    missing = set(FAMILIES) - set(families)
    if missing:
        raise ValueError(f"Missing family specs: {sorted(missing)}")
    for name, spec in families.items():
        if "pi_F" in spec or "kappa" in spec:
            raise ValueError(f"Forbidden key in family {name}")
    return families


def inverse_cdf(u: float, weights: Mapping[str, float]) -> str:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    threshold = float(u) * total
    cumulative = 0.0
    last_key = next(iter(weights))
    for key, weight in weights.items():
        last_key = key
        cumulative += float(weight)
        if threshold <= cumulative:
            return key
    return last_key


def scale(u: float, bounds: list[float] | tuple[float, float]) -> float:
    lo, hi = float(bounds[0]), float(bounds[1])
    return lo + float(u) * (hi - lo)


def conditional_alpha_bounds(family: str, variable: str, h_R: float, model_config: Mapping) -> list[float]:
    spec = model_config["alpha_bounds"][family]
    conditional_key = f"{variable}_conditional"
    if conditional_key in spec:
        for rule in spec[conditional_key]:
            if _eval_h_r_condition(rule["condition"], h_R):
                return rule["bounds"]
        raise ValueError(f"No conditional alpha bound matched {family} {variable} h_R={h_R}")
    return spec[variable]


def _eval_h_r_condition(condition: str, h_R: float) -> bool:
    safe = {"h_R": h_R, "abs": abs}
    return bool(eval(condition, {"__builtins__": {}}, safe))

