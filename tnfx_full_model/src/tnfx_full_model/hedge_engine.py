from __future__ import annotations

import numpy as np
import pandas as pd

from .forward_pricing import executable_terms
from .gamma_calibration import h_star_formula
from .market_data import join_market


def compute_hedge_decisions(
    handoff: pd.DataFrame,
    market_snapshot: pd.DataFrame,
    valuation_date: str,
    hedge_scenarios: dict[str, float],
    gamma_R_by_scenario: dict[str, float],
    gamma_override_by_scenario: dict[str, float] | None = None,
    gamma_source_label: str = "global_calibration",
    brown_toft_constant: float = 0.0,
    forward_bias: float = 0.0,
    min_abs_exposure: float = 1.0e-6,
    min_relative_exposure: float = 0.001,
) -> pd.DataFrame:
    joined = join_market(handoff, market_snapshot, valuation_date)
    rows = []
    for row in joined.to_dict("records"):
        terms = executable_terms(row, forward_bias=forward_bias)
        E = float(row["E_t"])
        sigma_E = max(float(row["sigma_E"]), 1e-12)
        delta_net_k = abs(float(row.get("delta_net_k", E)))
        exposure_abs = abs(E)
        materiality_threshold = max(float(min_abs_exposure), float(min_relative_exposure) * delta_net_k)
        is_material = exposure_abs >= materiality_threshold
        rho_safe = float(np.clip(float(row["rho"]), -0.999, 0.999))
        scale = (exposure_abs * float(terms["S0"])) ** 2
        for scenario, target in hedge_scenarios.items():
            gamma_map = gamma_override_by_scenario if gamma_override_by_scenario is not None else gamma_R_by_scenario
            lam = float(row["lambda"])
            row_status = "material" if is_material else "immaterial_exposure"
            if row_status == "immaterial_exposure":
                gamma_R = 0.0 if scenario == "no_hedge" else (np.nan if scenario == "full_hedge" else float(gamma_map.get(scenario, 1.0)))
                h_star = 0.0
                h_c = 0.0
                binding = "immaterial"
                variance_unhedged = 0.0
                variance_hedged = 0.0
                he = np.nan
                gamma_R_used = np.nan if scenario in {"no_hedge", "full_hedge"} else gamma_R
                gamma_R_source = "rule_based_no_hedge" if scenario == "no_hedge" else (
                    "rule_based_full_hedge" if scenario == "full_hedge" else gamma_source_label
                )
            elif scenario == "no_hedge":
                gamma_R = 0.0
                h_star = 0.0
                h_c = 0.0
                binding = "none"
                variance_unhedged = scale * (
                    sigma_E ** 2
                    + float(row["sigma_Q"]) ** 2
                    + 2.0 * rho_safe * sigma_E * float(row["sigma_Q"])
                )
                variance_hedged = variance_unhedged
                he = 0.0
                gamma_R_used = np.nan
                gamma_R_source = "rule_based_no_hedge"
            else:
                if scenario == "full_hedge":
                    gamma_R = np.nan
                    h_star = 1.0
                    h_c = min(1.0, lam)
                    gamma_R_used = np.nan
                    gamma_R_source = "rule_based_full_hedge"
                else:
                    gamma_R = float(gamma_map.get(scenario, 1.0))
                    h_star = h_star_formula(E, sigma_E, float(row["rho"]), float(row["sigma_Q"]), terms["S0"], terms["carry_cost"], gamma_R, forward_bias)
                    h_c = min(max(h_star, 0.0), lam)
                    gamma_R_used = gamma_R
                    gamma_R_source = gamma_source_label
                if h_star < 0:
                    binding = "lower"
                elif h_star > lam:
                    binding = "upper"
                else:
                    binding = "none"
                variance_unhedged = scale * (
                    sigma_E ** 2
                    + float(row["sigma_Q"]) ** 2
                    + 2.0 * rho_safe * sigma_E * float(row["sigma_Q"])
                )
                variance_hedged = scale * (
                    (1.0 - h_c) ** 2 * sigma_E ** 2
                    + float(row["sigma_Q"]) ** 2
                    + 2.0 * (1.0 - h_c) * rho_safe * sigma_E * float(row["sigma_Q"])
                )
                if variance_unhedged < 0 and abs(variance_unhedged) < 1e-18:
                    variance_unhedged = 0.0
                if variance_hedged < 0 and abs(variance_hedged) < 1e-18:
                    variance_hedged = 0.0
                he = 1.0 - variance_hedged / variance_unhedged if variance_unhedged > 0 else np.nan
                if (not np.isfinite(h_star)) or (abs(h_star) > 100):
                    row_status = "immaterial_exposure"
                    h_star = 0.0
                    h_c = 0.0
                    binding = "immaterial"
                    variance_unhedged = 0.0
                    variance_hedged = 0.0
                    he = np.nan
                    gamma_R_used = np.nan if scenario in {"no_hedge", "full_hedge"} else gamma_R
            expected_cost = h_c * exposure_abs * abs(terms["carry_cost"] + forward_bias) / terms["S0"]
            signed_carry_effect = h_c * exposure_abs * (terms["carry_cost"] + forward_bias) / terms["S0"]
            rows.append({
                "profile_id": row["profile_id"], "family": row["family"], "currency": row["currency"],
                "currency_pair": row["currency_pair"], "timing_cv_scenario": row["timing_cv_scenario"],
                "tenor_months": row["tenor_months"], "hedge_intensity_scenario": scenario,
                "target_intensity": float(target), "E_t": E, "direction": row["direction"],
                "delta_net_k": float(row.get("delta_net_k", E)), "materiality_threshold": materiality_threshold,
                "stage2_row_status": row_status,
                "lambda": lam, "sigma_E": sigma_E, "rho": row["rho"], "sigma_Q": row["sigma_Q"],
                "S0": terms["S0"], "F_executable": terms["F_executable"], "carry_cost": terms["carry_cost"],
                "forward_bias": forward_bias, "gamma_R": gamma_R, "h_star": h_star, "h_c": h_c,
                "gamma_R_used": gamma_R_used, "gamma_R_source": gamma_R_source,
                "binding_constraint": binding, "expected_cost": expected_cost, "signed_carry_effect": signed_carry_effect,
                "variance_unhedged": variance_unhedged, "variance_hedged": variance_hedged, "HE_t": he,
                "pricing_side": terms["pricing_side"], "valuation_date": valuation_date,
                "realized_future_date": row.get("realized_future_date"),
                "realized_future_spot_bid": row.get("realized_future_spot_bid"),
                "realized_future_spot_ask": row.get("realized_future_spot_ask"),
            })
    return pd.DataFrame(rows)


def aggregate_hedge_profile(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    grouped = decisions.groupby(["profile_id", "hedge_intensity_scenario"], as_index=False).agg(
        total_variance_unhedged=("variance_unhedged", "sum"),
        total_variance_hedged=("variance_hedged", "sum"),
        total_expected_cost=("expected_cost", "sum"),
        mean_h_c=("h_c", "mean"),
    )
    dom_cur = decisions.groupby(["profile_id", "hedge_intensity_scenario", "currency"])["E_t"].apply(lambda s: s.abs().sum()).reset_index()
    dom_cur = dom_cur.loc[dom_cur.groupby(["profile_id", "hedge_intensity_scenario"])["E_t"].idxmax()]
    dom_tenor = decisions.groupby(["profile_id", "hedge_intensity_scenario", "tenor_months"])["E_t"].apply(lambda s: s.abs().sum()).reset_index()
    dom_tenor = dom_tenor.loc[dom_tenor.groupby(["profile_id", "hedge_intensity_scenario"])["E_t"].idxmax()]
    mix = decisions.groupby(["profile_id", "hedge_intensity_scenario", "currency"])["E_t"].apply(lambda s: s.abs().sum()).reset_index()
    total = mix.groupby(["profile_id", "hedge_intensity_scenario"])["E_t"].transform("sum")
    mix["part"] = mix["currency"] + ":" + (mix["E_t"] / total).round(3).astype(str)
    mix = mix.groupby(["profile_id", "hedge_intensity_scenario"])["part"].apply(";".join).reset_index(name="currency_mix")
    return (grouped
            .merge(dom_cur[["profile_id", "hedge_intensity_scenario", "currency"]].rename(columns={"currency": "dominant_currency"}))
            .merge(dom_tenor[["profile_id", "hedge_intensity_scenario", "tenor_months"]].rename(columns={"tenor_months": "dominant_tenor"}))
            .merge(mix))


def cohort_analysis(decisions: pd.DataFrame, handoff: pd.DataFrame) -> pd.DataFrame:
    flags = ["g_PAE", "g_AEO", "g_ACC", "g_EXT", "g_CIRC"]
    enriched = decisions.merge(handoff[["profile_id", "currency_pair", "timing_cv_scenario", "tenor_months", *flags]].drop_duplicates(),
                               on=["profile_id", "currency_pair", "timing_cv_scenario", "tenor_months"], how="left")
    return enriched.groupby(["family", *flags, "hedge_intensity_scenario"], as_index=False).agg(
        mean_h_c=("h_c", "mean"), mean_expected_cost=("expected_cost", "mean"),
        mean_HE_t=("HE_t", "mean"), count_rows=("h_c", "size"),
    )
