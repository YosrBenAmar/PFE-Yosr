from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .constants import FAMILIES, REGIME_FLAGS, SOBOL_COLUMNS
from .currency_layer import decompose_currency
from .exposure_engine import conditional_h_c_bounds_for_family, exposure_diagnostics, family_sign_check
from .family_specs import conditional_alpha_bounds, inverse_cdf, load_families, scale
from .regime_engine import apply_regime_effects, select_regime
from .schemas import ACCEPTED_PROFILE_COLUMNS


class ConditionalIntervalError(ValueError):
    pass


def sobol_points(n: int, dimension: int, seed: int, burn_in: int) -> np.ndarray:
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pts = sampler.random(n + burn_in)
    return pts[burn_in:]


def target_family_counts(target_profiles: int, priors: dict[str, float]) -> dict[str, int]:
    raw = {family: int(round(target_profiles * float(weight))) for family, weight in priors.items()}
    diff = int(target_profiles) - sum(raw.values())
    order = sorted(priors, key=lambda f: float(priors[f]), reverse=True)
    idx = 0
    while diff != 0:
        family = order[idx % len(order)]
        if diff > 0:
            raw[family] += 1
            diff -= 1
        elif raw[family] > 0:
            raw[family] -= 1
            diff += 1
        idx += 1
    return raw


def _candidate_from_u(
    u: np.ndarray,
    sobol_index: int,
    prior_mode: str,
    model_config: dict,
    forced_family: str | None = None,
) -> dict:
    families = load_families(model_config)
    family = forced_family or inverse_cdf(float(u[13]), model_config["family_priors"][prior_mode])
    regime = select_regime(float(u[14]), family, model_config)
    spec = families[family]
    values = {
        "h_R": scale(u[0], spec["h_R"]),
        "r": scale(u[2], spec["r"]),
        "beta": scale(u[3], spec["beta"]),
        "fx_debt_service_share": scale(u[4], spec["fx_debt_service_share"]),
        "c": scale(u[8], spec["c"]),
        "lambda": scale(u[9], spec["lambda"]),
        "sigma_Q": scale(u[10], spec["sigma_Q"]),
        "rho": scale(u[11], spec["rho"]),
        "f": scale(u[12], spec["f"]),
        # f: frequency proxy for exposure crystallisation events.
        # Carried in Sobol dim 13 to preserve 15-D low-discrepancy structure.
        # Not consumed by Stage 1, 1.5, or Stage 2 computations.
        # Reserved for Stage 3 intra-period settlement frequency modeling.
    }
    values = apply_regime_effects(values, spec, regime, model_config)
    h_c_conditional = conditional_h_c_bounds_for_family(
        family, values["h_R"], values["r"], model_config["sign_thresholds"], spec["h_C"]
    )
    if h_c_conditional is None:
        values["h_C"] = scale(u[1], spec["h_C"])
    else:
        low, high = h_c_conditional
        if low > high:
            raise ConditionalIntervalError("conditional_h_C_interval_empty")
        values["h_C"] = scale(u[1], [low, high])
    values["alpha_R_EUR"] = scale(u[5], conditional_alpha_bounds(family, "alpha_R_EUR", values["h_R"], model_config))
    values["alpha_C_EUR"] = scale(u[6], conditional_alpha_bounds(family, "alpha_C_EUR", values["h_R"], model_config))
    values["alpha_D_EUR"] = scale(u[7], conditional_alpha_bounds(family, "alpha_D_EUR", values["h_R"], model_config))
    values["alpha_R_USD"] = 1.0 - values["alpha_R_EUR"]
    values["alpha_C_USD"] = 1.0 - values["alpha_C_EUR"]
    values["alpha_D_USD"] = 1.0 - values["alpha_D_EUR"]
    return {
        "sobol_index": sobol_index,
        "prior_mode": prior_mode,
        "family": family,
        "subtype": regime["regime_state"],
        **{f: int(regime[f]) for f in REGIME_FLAGS},
        **values,
    }


def _accept_candidate(candidate: dict, model_config: dict) -> tuple[bool, str, dict | None]:
    diag = exposure_diagnostics(candidate, model_config["sign_thresholds"])
    if not family_sign_check(candidate["family"], diag["delta_CF_total"], model_config["sign_thresholds"]):
        return False, "family_sign_validation_failed", None
    currency = decompose_currency(candidate)
    return True, "accepted", {**diag, **currency}


def _is_active(diag: dict, model_config: dict) -> bool:
    threshold = float(model_config.get("active_currency_threshold", 0.005))
    return abs(float(diag["delta_net_EUR"])) >= threshold or abs(float(diag["delta_net_USD"])) >= threshold


def _acceptance_status(rate: float, run_config: dict) -> str:
    if rate < run_config["acceptance_fail_floor"]:
        return "hard_fail"
    if rate < run_config["acceptance_hard_floor"]:
        return "validation_fail"
    if rate < run_config["acceptance_warning_floor"]:
        return "warning"
    return "pass"


def pilot_acceptance(model_config: dict, run_config: dict, prior_mode: str) -> pd.DataFrame:
    sampling_mode = run_config.get("sampling_mode", "pooled_sobol")
    rows = []
    families_to_run = FAMILIES if sampling_mode == "stratified_by_family" else [None]
    aggregate = {f: {"drawn": 0, "accepted": 0, "active_accepted": 0, "conditional_h_C_interval_empty": 0} for f in FAMILIES}
    for family_override in families_to_run:
        seed_offset = FAMILIES.index(family_override) * 10000 if family_override else 0
        points = sobol_points(run_config["pilot_points"], run_config["sobol_dimension"], run_config["seed"] + seed_offset, run_config["sobol_burn_in"])
        for i, u in enumerate(points):
            try:
                candidate = _candidate_from_u(u, run_config["sobol_burn_in"] + i, prior_mode, model_config, family_override)
            except ConditionalIntervalError:
                family = family_override or "unknown"
                if family in aggregate:
                    aggregate[family]["drawn"] += 1
                    aggregate[family]["conditional_h_C_interval_empty"] += 1
                continue
            family = candidate["family"]
            aggregate[family]["drawn"] += 1
            accepted, _, diag = _accept_candidate(candidate, model_config)
            aggregate[family]["accepted"] += int(accepted)
            aggregate[family]["active_accepted"] += int(accepted and diag is not None and _is_active(diag, model_config))
    for family, stats in aggregate.items():
        rate = stats["accepted"] / stats["drawn"] if stats["drawn"] else 0.0
        active_rate = stats["active_accepted"] / stats["drawn"] if stats["drawn"] else 0.0
        rows.append({"family": family, **stats, "acceptance_rate": rate, "active_acceptance_rate": active_rate, "status": _acceptance_status(rate, run_config)})
    return pd.DataFrame(rows)


def _profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ACCEPTED_PROFILE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[ACCEPTED_PROFILE_COLUMNS]


def _diagnostic_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "profile_id", "family", "delta_CF_total", "delta_profit_total", "profit_leverage_flag",
        "h_R_EUR", "h_R_USD", "h_C_EUR", "h_C_USD", "delta_CF_EUR", "delta_CF_USD",
        "delta_op_eff_EUR", "delta_op_eff_USD", "fx_debt_service_share_EUR",
        "fx_debt_service_share_USD", "delta_net_EUR", "delta_net_USD", "delta_net_total",
        "currency_mismatch_gap", "currency_mismatch_flag", "family_sign_check",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[cols]


def _add_sampling_weights(accepted: pd.DataFrame, prior_mode: str, model_config: dict) -> pd.DataFrame:
    accepted = accepted.copy()
    priors = model_config["family_priors"][prior_mode]
    shares = accepted["family"].value_counts(normalize=True).to_dict()
    accepted["target_prior_family_share"] = accepted["family"].map(lambda f: float(priors[f]))
    accepted["accepted_sample_family_share"] = accepted["family"].map(lambda f: float(shares.get(f, 0.0)))
    accepted["sampling_weight"] = accepted["target_prior_family_share"] / accepted["accepted_sample_family_share"].replace(0, np.nan)
    return accepted


def _sample_loop(
    model_config: dict,
    run_config: dict,
    prior_mode: str,
    target: int,
    family_override: str | None,
    profile_start: int,
    seed_offset: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], int]:
    accepted_rows, inactive_rows, diag_rows, reject_rows = [], [], [], []
    next_profile_id = profile_start
    n_candidates = max(int(math.ceil(target * 1.35)), int(run_config["pilot_points"]))
    chunk = 0
    while len(accepted_rows) < target:
        points = sobol_points(n_candidates, run_config["sobol_dimension"], run_config["seed"] + seed_offset + chunk, run_config["sobol_burn_in"])
        for i, u in enumerate(points):
            sobol_index = run_config["sobol_burn_in"] + seed_offset + chunk * n_candidates + i
            try:
                candidate = _candidate_from_u(u, sobol_index, prior_mode, model_config, family_override)
            except ConditionalIntervalError as exc:
                reject_rows.append({
                    "sobol_index": sobol_index, "prior_mode": prior_mode, "family_drawn": family_override or "pooled",
                    "regime_state_drawn": None, "accepted": False, "rejection_reason": str(exc),
                })
                continue
            accepted, reason, diag = _accept_candidate(candidate, model_config)
            reject_rows.append({
                "sobol_index": sobol_index, "prior_mode": prior_mode, "family_drawn": candidate["family"],
                "regime_state_drawn": candidate["subtype"], "accepted": accepted, "rejection_reason": reason,
            })
            if not accepted:
                continue
            candidate["profile_id"] = next_profile_id
            diag["profile_id"] = next_profile_id
            if not _is_active(diag, model_config):
                inactive = {**candidate, "inactive_reason": "no_active_currency"}
                inactive_rows.append(inactive)
                reject_rows[-1]["accepted"] = False
                reject_rows[-1]["rejection_reason"] = "no_active_currency"
                next_profile_id += 1
                continue
            accepted_rows.append(candidate)
            diag_rows.append(diag)
            next_profile_id += 1
            if len(accepted_rows) >= target:
                break
        chunk += 1
        n_candidates = max(n_candidates, target)
    return accepted_rows, inactive_rows, diag_rows, reject_rows, next_profile_id


def sample_profiles(model_config: dict, run_config: dict, prior_mode: str | None = None) -> dict[str, pd.DataFrame]:
    prior_mode = prior_mode or run_config["primary_prior_mode"]
    pilot = pilot_acceptance(model_config, run_config, prior_mode)
    target = int(run_config["target_profiles"])
    sampling_mode = run_config.get("sampling_mode", "pooled_sobol")
    accepted_rows, inactive_rows, diag_rows, reject_rows = [], [], [], []
    next_profile_id = 1
    if sampling_mode == "stratified_by_family":
        counts = target_family_counts(target, model_config["family_priors"][prior_mode])
        for offset, family in enumerate(FAMILIES):
            rows, inactive, diag, rejects, next_profile_id = _sample_loop(
                model_config, run_config, prior_mode, counts[family], family, next_profile_id, (offset + 1) * 100000
            )
            accepted_rows.extend(rows)
            inactive_rows.extend(inactive)
            diag_rows.extend(diag)
            reject_rows.extend(rejects)
    elif sampling_mode == "pooled_sobol":
        rows, inactive, diag, rejects, next_profile_id = _sample_loop(
            model_config, run_config, prior_mode, target, None, next_profile_id, 0
        )
        accepted_rows.extend(rows)
        inactive_rows.extend(inactive)
        diag_rows.extend(diag)
        reject_rows.extend(rejects)
    else:
        raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")
    accepted = _add_sampling_weights(pd.DataFrame(accepted_rows), prior_mode, model_config)
    accepted = _profile_columns(accepted)
    diagnostics = _diagnostic_columns(pd.DataFrame(diag_rows))
    inactive = pd.DataFrame(inactive_rows)
    if not inactive.empty:
        inactive = _add_sampling_weights(inactive, prior_mode, model_config)
    return {
        "Accepted_Profiles": accepted,
        "Inactive_Profiles": inactive,
        "BM_Exposure_Diagnostics": diagnostics,
        "Rejection_Log": pd.DataFrame(reject_rows),
        "Sobol_Acceptance": pilot,
        "Sobol_Meta": pd.DataFrame([{
            "seed": run_config["seed"],
            "sobol_dimension": run_config["sobol_dimension"],
            "sobol_burn_in": run_config["sobol_burn_in"],
            "first_points_skipped": run_config["sobol_burn_in"],
            "sampling_mode": sampling_mode,
            "prior_mode": prior_mode,
        }]),
    }


def sobol_columns() -> list[str]:
    return SOBOL_COLUMNS.copy()
