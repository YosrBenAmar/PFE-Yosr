from __future__ import annotations

import numpy as np
import pandas as pd


def _sample_profiles(accepted_profiles: pd.DataFrame, profile_sample_size: int, seed: int) -> pd.DataFrame:
    family_priors = {"importer": 0.35, "exporter": 0.20, "processor": 0.30, "trader": 0.15}
    rng = np.random.default_rng(seed)
    parts = []
    for family, prior in family_priors.items():
        target = int(round(profile_sample_size * prior))
        fam = accepted_profiles[accepted_profiles["family"] == family].copy()
        if fam.empty or target <= 0:
            continue
        replace = len(fam) < target
        choice = rng.choice(fam.index.to_numpy(), size=target, replace=replace)
        parts.append(fam.loc[choice].copy())
    sample = pd.concat(parts, ignore_index=True) if parts else accepted_profiles.head(0).copy()
    if len(sample) > profile_sample_size:
        sample = sample.sample(n=profile_sample_size, random_state=seed).copy()
    elif len(sample) < profile_sample_size and not accepted_profiles.empty:
        remaining = profile_sample_size - len(sample)
        replace = len(accepted_profiles) < remaining
        extra = accepted_profiles.sample(n=remaining, replace=replace, random_state=seed).copy()
        sample = pd.concat([sample, extra], ignore_index=True)
    return sample.reset_index(drop=True)


def compute_profile_cohort_attribution(
    rolling_market_performance: pd.DataFrame,
    accepted_profiles: pd.DataFrame,
    stage_1_5_handoff: pd.DataFrame,
    profile_sample_size: int = 1000,
    stratify_by: list[str] = ["family"],
    seed: int = 42,
    run_id: str | None = None,
) -> pd.DataFrame:
    if accepted_profiles.empty or stage_1_5_handoff.empty or rolling_market_performance.empty:
        return pd.DataFrame(columns=[
            "run_id", "family", "regime_label", "hedge_intensity_scenario", "forward_stress_scenario",
            "currency_pair", "tenor_months", "n_profiles", "n_rolling_observations", "mean_h_c_capped",
            "mean_profile_realized_pnl", "mean_HE_t", "hit_ratio", "pnl_5pct_quantile", "pnl_95pct_quantile",
            "worst_case_pnl", "best_case_pnl",
        ])

    sample = _sample_profiles(accepted_profiles, profile_sample_size, seed)
    if sample.empty:
        return pd.DataFrame()

    handoff = stage_1_5_handoff.copy()
    if "lambda" not in handoff.columns:
        raise ValueError("Stage_1_5_Handoff is missing lambda column")
    handoff = handoff.merge(sample[["profile_id", "family", "lambda"]].drop_duplicates(), on=["profile_id", "family"], how="inner")
    handoff = handoff.copy()
    handoff["abs_E_t"] = pd.to_numeric(handoff["E_t"], errors="coerce").abs()

    handoff_summary = handoff.groupby(["family", "currency_pair", "tenor_months"], sort=False).agg(
        n_profiles=("profile_id", "nunique"),
        mean_lambda=("lambda", "mean"),
        mean_abs_E_t=("abs_E_t", "mean"),
        q05_abs_E_t=("abs_E_t", lambda s: float(s.quantile(0.05))),
        q95_abs_E_t=("abs_E_t", lambda s: float(s.quantile(0.95))),
        min_abs_E_t=("abs_E_t", "min"),
        max_abs_E_t=("abs_E_t", "max"),
    ).reset_index()

    roll_summary = rolling_market_performance.groupby(["regime_label", "hedge_intensity_scenario", "forward_stress_scenario", "currency_pair", "tenor_months"], sort=False).agg(
        n_roll_rows=("h_c", "size"),
        mean_h_c=("h_c", "mean"),
        mean_forward_advantage=("forward_advantage", "mean"),
        mean_HE_t=("HE_t", "mean"),
        hit_ratio=("forward_advantage", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
        pnl_5pct_quantile=("forward_advantage", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.05))),
        pnl_95pct_quantile=("forward_advantage", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.95))),
        worst_case_pnl=("forward_advantage", "min"),
        best_case_pnl=("forward_advantage", "max"),
    ).reset_index()

    output_rows = []
    for _, hand_row in handoff_summary.iterrows():
        subset = roll_summary[(roll_summary["currency_pair"] == hand_row["currency_pair"]) & (roll_summary["tenor_months"] == hand_row["tenor_months"])]
        for _, roll_row in subset.iterrows():
            capped_h = min(float(roll_row["mean_h_c"]), float(hand_row["mean_lambda"]))
            output_rows.append({
                "family": hand_row["family"],
                "regime_label": roll_row["regime_label"],
                "hedge_intensity_scenario": roll_row["hedge_intensity_scenario"],
                "forward_stress_scenario": roll_row["forward_stress_scenario"],
                "currency_pair": hand_row["currency_pair"],
                "tenor_months": int(hand_row["tenor_months"]),
                "n_profiles": int(hand_row["n_profiles"]),
                "n_rolling_observations": int(hand_row["n_profiles"] * roll_row["n_roll_rows"]),
                "mean_h_c_capped": capped_h,
                "mean_profile_realized_pnl": float(capped_h * float(roll_row["mean_forward_advantage"]) * float(hand_row["mean_abs_E_t"])),
                "mean_HE_t": float(roll_row["mean_HE_t"]),
                "hit_ratio": float(roll_row["hit_ratio"]),
                "pnl_5pct_quantile": float(float(roll_row["pnl_5pct_quantile"]) * float(hand_row["q05_abs_E_t"])),
                "pnl_95pct_quantile": float(float(roll_row["pnl_95pct_quantile"]) * float(hand_row["q95_abs_E_t"])),
                "worst_case_pnl": float(float(roll_row["worst_case_pnl"]) * float(hand_row["max_abs_E_t"])),
                "best_case_pnl": float(float(roll_row["best_case_pnl"]) * float(hand_row["max_abs_E_t"])),
            })

    result = pd.DataFrame(output_rows)
    if run_id is not None and not result.empty:
        result.insert(0, "run_id", run_id)
    return result