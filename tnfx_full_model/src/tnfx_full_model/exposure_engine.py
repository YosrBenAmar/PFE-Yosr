from __future__ import annotations

from typing import Mapping


def compute_delta_cf_total(profile: Mapping) -> float:
    return float(profile["h_R"]) - float(profile["h_C"]) * (1.0 - float(profile["r"]))


def family_sign_check(family: str, delta_cf_total: float, thresholds: Mapping) -> bool:
    if family == "importer":
        return delta_cf_total <= -float(thresholds["epsilon_I"])
    if family == "exporter":
        return delta_cf_total >= float(thresholds["epsilon_E"])
    if family == "processor":
        return abs(delta_cf_total) <= float(thresholds["tau_P"])
    if family == "trader":
        return abs(delta_cf_total) <= float(thresholds["tau_T"])
    raise ValueError(f"Unknown family {family}")


def balanced_h_c_interval(h_R: float, r: float, tau: float, h_c_bounds: list[float] | tuple[float, float]) -> tuple[float, float]:
    denominator = 1.0 - float(r)
    if denominator <= 0:
        raise ZeroDivisionError("Cannot compute conditional h_C interval when r >= 1")
    implied_low = (float(h_R) - float(tau)) / denominator
    implied_high = (float(h_R) + float(tau)) / denominator
    low = max(float(h_c_bounds[0]), implied_low)
    high = min(float(h_c_bounds[1]), implied_high)
    return low, high


def conditional_h_c_bounds_for_family(family: str, h_R: float, r: float, thresholds: Mapping, h_c_bounds: list[float] | tuple[float, float]) -> tuple[float, float] | None:
    if family == "processor":
        return balanced_h_c_interval(h_R, r, thresholds["tau_P"], h_c_bounds)
    if family == "trader":
        return balanced_h_c_interval(h_R, r, thresholds["tau_T"], h_c_bounds)
    return None


def exposure_diagnostics(profile: Mapping, thresholds: Mapping) -> dict:
    delta = compute_delta_cf_total(profile)
    r = float(profile["r"])
    profit = delta / r
    return {
        "profile_id": profile.get("profile_id"),
        "family": profile["family"],
        "delta_CF_total": delta,
        "delta_profit_total": profit,
        "profit_leverage_flag": abs(profit) > 10.0,
        "family_sign_check": family_sign_check(profile["family"], delta, thresholds),
    }
