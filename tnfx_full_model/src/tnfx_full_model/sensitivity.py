from __future__ import annotations

import pandas as pd

from .exposure_engine import family_sign_check


def sign_threshold_sensitivity(diagnostics: pd.DataFrame, model_config: dict) -> pd.DataFrame:
    rows = []
    mapping = {
        "importer": "epsilon_I", "exporter": "epsilon_E", "processor": "tau_P", "trader": "tau_T",
    }
    for family, threshold_name in mapping.items():
        subset = diagnostics[diagnostics["family"] == family]
        for value in model_config["sign_threshold_sensitivity"][threshold_name]:
            thresholds = dict(model_config["sign_thresholds"])
            thresholds[threshold_name] = value
            rate = subset["delta_CF_total"].apply(lambda x: family_sign_check(family, x, thresholds)).mean() if not subset.empty else 0.0
            rows.append({"family": family, "threshold_name": threshold_name, "threshold_value": value,
                         "acceptance_rate": rate, "rejection_rate": 1.0 - rate})
    return pd.DataFrame(rows)


def macro_anchor_check(diagnostics: pd.DataFrame, model_config: dict) -> pd.DataFrame:
    means = diagnostics.groupby("family")["delta_CF_total"].mean().to_dict()
    lower, upper = model_config["macro_anchor"]["lower"], model_config["macro_anchor"]["upper"]
    provisional = []
    for prior_mode, weights in model_config["family_priors"].items():
        weighted = sum(float(weights[f]) * float(means.get(f, 0.0)) for f in weights)
        provisional.append({
            "prior_mode": prior_mode, "importer_weight": weights["importer"], "exporter_weight": weights["exporter"],
            "processor_weight": weights["processor"], "trader_weight": weights["trader"],
            "weighted_delta_CF_total": weighted, "lower_bound": lower, "upper_bound": upper,
            "pass_fail": "pass" if lower <= weighted <= upper else "fail",
        })
    nominal = next((r for r in provisional if r["prior_mode"] == "nominal"), None)
    nominal_passes = nominal is not None and nominal["pass_fail"] == "pass"
    all_pass = all(r["pass_fail"] == "pass" for r in provisional)
    if all_pass:
        status = "all_priors_pass"
        interpretation = "The synthetic population passes the macro-anchor corridor under all configured family priors."
    elif nominal_passes:
        status = "nominal_pass_only_prior_sensitive"
        interpretation = "The synthetic population is calibrated under the nominal prior. Alternative priors are treated as sensitivity views and indicate prior dependence."
    else:
        status = "nominal_fail_recalibration_required"
        interpretation = "The nominal calibration fails the macro-anchor corridor and requires manual recalibration."
    rows = []
    for row in provisional:
        rows.append({**row, "macro_anchor_status": status, "interpretation": interpretation})
    return pd.DataFrame(rows)


def sensitivity_summary(model_config: dict, market_config: dict, gamma_detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, cv in model_config["timing_cv_scenarios"].items():
        rows.append({"panel": "timing_CV", "item": name, "value": cv, "status": "implemented"})
    for name, target in market_config["hedge_intensity_scenarios"].items():
        rows.append({"panel": "hedge_intensity", "item": name, "value": target, "status": "implemented"})
    for mode in market_config["forward_bias_modes"]:
        rows.append({"panel": "forward_bias", "item": mode, "value": "", "status": "implemented" if mode == "unbiased" else "placeholder"})
    rows.extend([
        {"panel": "vol_spec", "item": "rolling", "value": "252 observations", "status": "implemented"},
        {"panel": "vol_spec", "item": "GARCH", "value": "", "status": "future_placeholder_not_implemented"},
        # CIP wedge is currently computed in Market_Data_Snapshot as carry_cost_wedge.
        # It is not yet wired through the hedge optimizer as an alternative decision run.
        {"panel": "CIP", "item": "CIP+50bps", "value": market_config["cip_wedge_bps"], "status": "computed_in_snapshot_not_consumed_by_optimizer"},
        {"panel": "alpha_bounds", "item": "anchored", "value": "", "status": "implemented"},
        {"panel": "alpha_bounds", "item": "uniform_[0,1]", "value": "", "status": "placeholder_if_configured"},
        {"panel": "CVaR objective", "item": "CVaR", "value": "", "status": "future_placeholder_not_implemented"},
    ])
    if gamma_detail is not None and not gamma_detail.empty:
        warn = bool(gamma_detail["robustness_warning"].fillna(False).any())
        rows.append({"panel": "gamma_R", "item": "cross_config_robustness", "value": "", "status": "warning" if warn else "implemented"})
    return pd.DataFrame(rows)
