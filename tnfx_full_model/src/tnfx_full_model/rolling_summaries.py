from __future__ import annotations

import numpy as np
import pandas as pd


def _summary_base(rolling_market_performance: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    if rolling_market_performance.empty:
        return pd.DataFrame(columns=group_keys)
    df = rolling_market_performance.copy()
    df["hedge_spot_rate"] = pd.to_numeric(df["hedge_spot_rate"], errors="coerce")
    grouped = df.groupby(group_keys, as_index=False).agg(
        n_observations=("forward_advantage", "size"),
        hedge_spot_rate_avg=("hedge_spot_rate", "mean"),
        mean_forward_advantage=("forward_advantage", "mean"),
        median_forward_advantage=("forward_advantage", "median"),
        hit_ratio=("hit_indicator", "mean"),
        mean_hedged_pnl_per_unit=("hedged_pnl_per_unit", "mean"),
        std_hedged_pnl_per_unit=("hedged_pnl_per_unit", "std"),
        worst_case_pnl=("hedged_pnl_per_unit", "min"),
        best_case_pnl=("hedged_pnl_per_unit", "max"),
        mean_HE_t=("HE_t", "mean"),
        median_HE_t=("HE_t", "median"),
        share_negative_HE=("HE_t", lambda s: float((pd.to_numeric(s, errors="coerce") < 0).mean())),
    )
    grouped["mean_forward_advantage_bps"] = grouped["mean_forward_advantage"] * 10000.0 / grouped["hedge_spot_rate_avg"].replace(0, np.nan)
    return grouped.drop(columns=["hedge_spot_rate_avg"])


def compute_rolling_market_summary(rolling_market_performance: pd.DataFrame) -> pd.DataFrame:
    return _summary_base(rolling_market_performance, ["currency_pair", "side", "tenor_months", "hedge_intensity_scenario", "forward_stress_scenario"])


def compute_regime_performance(rolling_market_performance: pd.DataFrame) -> pd.DataFrame:
    return _summary_base(rolling_market_performance, ["currency_pair", "side", "tenor_months", "regime_label", "hedge_intensity_scenario", "forward_stress_scenario"])


def compute_strategy_ranking(rolling_market_performance: pd.DataFrame) -> pd.DataFrame:
    if rolling_market_performance.empty:
        return pd.DataFrame(columns=[
            "currency_pair", "side", "tenor_months", "regime_label", "forward_stress_scenario",
            "hedge_intensity_scenario", "rank", "is_recommended", "recommendation_reason",
        ])
    summary = _summary_base(rolling_market_performance, ["currency_pair", "side", "tenor_months", "regime_label", "forward_stress_scenario", "hedge_intensity_scenario"])
    ranked = summary.sort_values(
        ["currency_pair", "side", "tenor_months", "regime_label", "forward_stress_scenario", "hit_ratio", "mean_hedged_pnl_per_unit", "std_hedged_pnl_per_unit", "mean_HE_t"],
        ascending=[True, True, True, True, True, False, False, True, False],
    ).copy()
    ranked["rank"] = ranked.groupby(["currency_pair", "side", "tenor_months", "regime_label", "forward_stress_scenario"]).cumcount() + 1
    ranked["is_recommended"] = ranked["rank"].eq(1)
    top = ranked[ranked["is_recommended"]].copy()
    top["recommendation_reason"] = top.apply(lambda r: f"highest hit ratio of {r['hit_ratio']:.2f} with mean P&L of {r['mean_hedged_pnl_per_unit']:.3f}", axis=1)
    ranked = ranked.merge(top[["currency_pair", "side", "tenor_months", "regime_label", "forward_stress_scenario", "recommendation_reason"]], on=["currency_pair", "side", "tenor_months", "regime_label", "forward_stress_scenario"], how="left")
    return ranked[["currency_pair", "side", "tenor_months", "regime_label", "forward_stress_scenario", "hedge_intensity_scenario", "rank", "is_recommended", "recommendation_reason"]]


def compute_negative_he_diagnostics(rolling_market_performance: pd.DataFrame) -> pd.DataFrame:
    if rolling_market_performance.empty:
        return pd.DataFrame(columns=[
            "currency_pair", "side", "tenor_months", "hedge_intensity_scenario", "forward_stress_scenario", "regime_label",
            "n_rows", "share_of_total_in_cell", "mean_HE_t", "min_HE_t", "mean_carry_cost_used", "mean_forward_advantage", "interpretation",
        ])
    df = rolling_market_performance.copy()
    total = df.groupby(["currency_pair", "side", "tenor_months", "hedge_intensity_scenario", "forward_stress_scenario", "regime_label"]).size().reset_index(name="total_rows")
    neg = df[df["HE_t"] < 0].groupby(["currency_pair", "side", "tenor_months", "hedge_intensity_scenario", "forward_stress_scenario", "regime_label"], as_index=False).agg(
        n_rows=("HE_t", "size"),
        mean_HE_t=("HE_t", "mean"),
        min_HE_t=("HE_t", "min"),
        mean_carry_cost_used=("carry_cost_used", "mean"),
        mean_forward_advantage=("forward_advantage", "mean"),
    )
    out = neg.merge(total, on=["currency_pair", "side", "tenor_months", "hedge_intensity_scenario", "forward_stress_scenario", "regime_label"], how="left")
    out["share_of_total_in_cell"] = out["n_rows"] / out["total_rows"].replace(0, np.nan)
    out["interpretation"] = "natural_offset_or_cost_dominates"
    return out.drop(columns=["total_rows"])