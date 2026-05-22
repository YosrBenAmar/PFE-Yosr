from __future__ import annotations

import numpy as np
import pandas as pd


def _minmax_norm(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    mask = values.notna()
    if not mask.any():
        return out
    subset = values[mask]
    lo = float(subset.min())
    hi = float(subset.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) <= 1e-12:
        out.loc[mask] = 0.5
        return out
    out.loc[mask] = (subset - lo) / (hi - lo)
    return out


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
            "corporate_hedging_score", "protection_score_component", "downside_score_component",
            "volatility_penalty_component", "pnl_score_component", "ranking_methodology",
            "hit_ratio", "mean_hedged_pnl_per_unit", "std_hedged_pnl_per_unit", "worst_case_pnl",
            "mean_HE_t", "median_HE_t", "share_negative_HE",
        ])
    group_cols = ["currency_pair", "side", "tenor_months", "regime_label", "forward_stress_scenario"]
    summary = _summary_base(rolling_market_performance, [*group_cols, "hedge_intensity_scenario"]).copy()
    for col in [
        "mean_HE_t", "median_HE_t", "share_negative_HE", "worst_case_pnl",
        "std_hedged_pnl_per_unit", "mean_hedged_pnl_per_unit", "hit_ratio",
    ]:
        if col not in summary.columns:
            summary[col] = np.nan
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    summary["median_HE_t"] = summary["median_HE_t"].fillna(summary["mean_HE_t"])
    summary["mean_HE_t_norm"] = summary.groupby(group_cols)["mean_HE_t"].transform(_minmax_norm)
    summary["median_HE_t_norm"] = summary.groupby(group_cols)["median_HE_t"].transform(_minmax_norm)
    summary["share_negative_HE_norm"] = summary.groupby(group_cols)["share_negative_HE"].transform(_minmax_norm)
    summary["worst_case_pnl_norm"] = summary.groupby(group_cols)["worst_case_pnl"].transform(_minmax_norm)
    summary["std_norm"] = summary.groupby(group_cols)["std_hedged_pnl_per_unit"].transform(_minmax_norm)
    summary["mean_pnl_norm"] = summary.groupby(group_cols)["mean_hedged_pnl_per_unit"].transform(_minmax_norm)
    summary["hit_ratio_norm"] = summary.groupby(group_cols)["hit_ratio"].transform(_minmax_norm)
    summary["protection_score_component"] = 0.70 * summary["mean_HE_t_norm"] + 0.30 * summary["median_HE_t_norm"]
    summary["downside_score_component"] = 0.60 * summary["worst_case_pnl_norm"] + 0.40 * (1.0 - summary["share_negative_HE_norm"])
    summary["volatility_penalty_component"] = summary["std_norm"]
    summary["pnl_score_component"] = 0.60 * summary["hit_ratio_norm"] + 0.40 * summary["mean_pnl_norm"]
    summary["corporate_hedging_score"] = (
        0.50 * summary["protection_score_component"]
        + 0.30 * summary["downside_score_component"]
        - 0.15 * summary["volatility_penalty_component"]
        + 0.05 * summary["pnl_score_component"]
    )
    summary["ranking_methodology"] = "corporate_protection_downside_volatility_cost_v1"
    ranked = summary.sort_values(
        [
            *group_cols,
            "corporate_hedging_score",
            "protection_score_component",
            "downside_score_component",
            "std_hedged_pnl_per_unit",
            "hit_ratio",
            "mean_hedged_pnl_per_unit",
            "hedge_intensity_scenario",
        ],
        ascending=[True, True, True, True, True, False, False, False, True, False, False, True],
    ).copy()
    ranked["rank"] = ranked.groupby(group_cols).cumcount() + 1
    ranked["is_recommended"] = ranked["rank"].eq(1)
    ranked["recommendation_reason"] = ranked.apply(
        lambda r: (
            f"corporate score={r['corporate_hedging_score']:.4f}; "
            f"protection={r['protection_score_component']:.4f}; "
            f"downside={r['downside_score_component']:.4f}; "
            f"volatility_penalty={r['volatility_penalty_component']:.4f}; "
            f"pnl={r['pnl_score_component']:.4f}"
        ),
        axis=1,
    )
    return ranked[[
        "currency_pair", "side", "tenor_months", "regime_label", "forward_stress_scenario",
        "hedge_intensity_scenario", "rank", "is_recommended", "recommendation_reason",
        "corporate_hedging_score", "protection_score_component", "downside_score_component",
        "volatility_penalty_component", "pnl_score_component", "ranking_methodology",
        "hit_ratio", "mean_hedged_pnl_per_unit", "std_hedged_pnl_per_unit", "worst_case_pnl",
        "mean_HE_t", "median_HE_t", "share_negative_HE",
    ]]


def compute_hedge_decision_recommendations(
    stage2_decisions: pd.DataFrame,
    strategy_ranking: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "profile_id", "family", "currency", "currency_pair", "tenor_months", "timing_cv_scenario", "direction",
        "E_t", "abs_exposure", "lambda", "selected_hedge_intensity_scenario", "recommended_hedge_ratio",
        "recommended_hedged_amount", "expected_cost", "mean_HE_t", "hit_ratio", "mean_hedged_pnl_per_unit",
        "recommendation_reason", "recommendation_source",
    ]
    if not isinstance(stage2_decisions, pd.DataFrame) or stage2_decisions.empty:
        return pd.DataFrame(columns=cols)
    dec = stage2_decisions.copy()
    required = [
        "profile_id", "family", "currency", "currency_pair", "tenor_months", "timing_cv_scenario",
        "direction", "E_t", "lambda", "h_c", "hedge_intensity_scenario", "expected_cost",
    ]
    missing = [c for c in required if c not in dec.columns]
    if missing:
        return pd.DataFrame(columns=cols)
    dec["E_t"] = pd.to_numeric(dec["E_t"], errors="coerce")
    dec["lambda"] = pd.to_numeric(dec["lambda"], errors="coerce")
    dec["h_c"] = pd.to_numeric(dec["h_c"], errors="coerce")
    dec["expected_cost"] = pd.to_numeric(dec["expected_cost"], errors="coerce")
    dec["HE_t"] = pd.to_numeric(dec.get("HE_t", np.nan), errors="coerce")
    if "pricing_side" in dec.columns:
        dec["side"] = dec["pricing_side"].astype(str).str.upper()
    else:
        dec["side"] = np.where(
            dec["direction"].astype(str).str.lower().eq("outflow"),
            "ASK",
            np.where(dec["direction"].astype(str).str.lower().eq("inflow"), "BID", ""),
        )
    rank_pick = pd.DataFrame()
    if isinstance(strategy_ranking, pd.DataFrame) and not strategy_ranking.empty:
        rank = strategy_ranking.copy()
        needed = {"currency_pair", "side", "tenor_months", "hedge_intensity_scenario"}
        if needed.issubset(rank.columns):
            if "forward_stress_scenario" in rank.columns:
                base = rank[rank["forward_stress_scenario"].astype(str).eq("cip_base")]
                if not base.empty:
                    rank = base
            if "corporate_hedging_score" not in rank.columns:
                rank["corporate_hedging_score"] = -pd.to_numeric(rank.get("rank", np.nan), errors="coerce").fillna(0.0)
            metric_cols = ["mean_HE_t", "hit_ratio", "mean_hedged_pnl_per_unit", "corporate_hedging_score"]
            for col in metric_cols:
                rank[col] = pd.to_numeric(rank.get(col, np.nan), errors="coerce")
            g1 = ["currency_pair", "side", "tenor_months", "hedge_intensity_scenario"]
            agg = rank.groupby(g1, as_index=False).agg(
                corporate_hedging_score=("corporate_hedging_score", "mean"),
                mean_HE_t=("mean_HE_t", "mean"),
                hit_ratio=("hit_ratio", "mean"),
                mean_hedged_pnl_per_unit=("mean_hedged_pnl_per_unit", "mean"),
                recommendation_reason=("recommendation_reason", "first"),
            )
            agg = agg.sort_values(
                ["currency_pair", "side", "tenor_months", "corporate_hedging_score", "mean_HE_t", "hit_ratio", "mean_hedged_pnl_per_unit", "hedge_intensity_scenario"],
                ascending=[True, True, True, False, False, False, False, True],
            )
            agg["__rank"] = agg.groupby(["currency_pair", "side", "tenor_months"]).cumcount() + 1
            rank_pick = agg[agg["__rank"] == 1][[
                "currency_pair", "side", "tenor_months", "hedge_intensity_scenario", "mean_HE_t",
                "hit_ratio", "mean_hedged_pnl_per_unit", "recommendation_reason",
            ]].rename(columns={"hedge_intensity_scenario": "selected_from_ranking"})
    join_cols = ["currency_pair", "side", "tenor_months"]
    if not rank_pick.empty:
        dec = dec.merge(rank_pick, on=join_cols, how="left")
    else:
        dec["selected_from_ranking"] = np.nan
        dec["recommendation_reason"] = np.nan
        dec["hit_ratio"] = np.nan
        dec["mean_hedged_pnl_per_unit"] = np.nan
        dec["mean_HE_t"] = np.nan
    dec["selected_by_ranking"] = dec["hedge_intensity_scenario"].astype(str) == dec["selected_from_ranking"].astype(str)
    key_cols = ["profile_id", "currency_pair", "tenor_months", "timing_cv_scenario", "direction"]
    dec["fallback_he_norm"] = dec.groupby(key_cols)["HE_t"].transform(_minmax_norm)
    dec["fallback_cost_norm"] = dec.groupby(key_cols)["expected_cost"].transform(_minmax_norm)
    dec["fallback_score"] = 0.75 * dec["fallback_he_norm"] + 0.25 * (1.0 - dec["fallback_cost_norm"])
    dec["__priority"] = np.where(dec["selected_by_ranking"], 1, 0)
    dec = dec.sort_values(
        key_cols + ["__priority", "fallback_score", "HE_t", "expected_cost", "hedge_intensity_scenario"],
        ascending=[True, True, True, True, True, False, False, False, True, True],
    )
    chosen = dec.drop_duplicates(key_cols, keep="first").copy()
    chosen["selected_hedge_intensity_scenario"] = chosen["hedge_intensity_scenario"]
    chosen["recommended_hedge_ratio"] = pd.to_numeric(chosen["h_c"], errors="coerce")
    chosen["abs_exposure"] = pd.to_numeric(chosen["E_t"], errors="coerce").abs()
    chosen["recommended_hedged_amount"] = chosen["recommended_hedge_ratio"] * chosen["abs_exposure"]
    chosen["recommendation_source"] = np.where(chosen["selected_by_ranking"], "strategy_ranking", "stage2_fallback")
    chosen["recommendation_reason"] = np.where(
        chosen["selected_by_ranking"] & chosen["recommendation_reason"].notna(),
        chosen["recommendation_reason"].astype(str),
        chosen.apply(
            lambda r: (
                f"fallback score={r['fallback_score']:.4f}; "
                f"HE_t={float(r['HE_t']) if pd.notna(r['HE_t']) else float('nan'):.4f}; "
                f"expected_cost={float(r['expected_cost']) if pd.notna(r['expected_cost']) else float('nan'):.6f}"
            ),
            axis=1,
        ),
    )
    chosen["mean_HE_t"] = np.where(
        chosen["selected_by_ranking"] & pd.to_numeric(chosen["mean_HE_t"], errors="coerce").notna(),
        pd.to_numeric(chosen["mean_HE_t"], errors="coerce"),
        pd.to_numeric(chosen["HE_t"], errors="coerce"),
    )
    chosen["hit_ratio"] = pd.to_numeric(chosen.get("hit_ratio", np.nan), errors="coerce")
    chosen["mean_hedged_pnl_per_unit"] = pd.to_numeric(chosen.get("mean_hedged_pnl_per_unit", np.nan), errors="coerce")
    return chosen[cols].reset_index(drop=True)


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
