from __future__ import annotations

import numpy as np
import pandas as pd

from .gamma_calibration import solve_gamma_for_target


DEFAULT_REGIME_BREAKS = {
    "pre_revolution_and_managed_crawl": {"start": "2007-09-12", "end": "2013-12-31"},
    "post_revolution_depreciation": {"start": "2014-01-01", "end": "2018-12-31"},
    "flexible_managed_regime": {"start": "2019-01-01", "end": "2026-12-31"},
}

REGIME_BREAKS = dict(DEFAULT_REGIME_BREAKS)


def set_regime_breaks(regime_breaks: dict | None) -> None:
    global REGIME_BREAKS
    if regime_breaks:
        REGIME_BREAKS = dict(regime_breaks)


def _assign_regime_label(dates: pd.Series) -> pd.Series:
    labels = pd.Series(np.full(len(dates), "unmapped", dtype=object), index=dates.index)
    for name, bounds in REGIME_BREAKS.items():
        start = pd.to_datetime(bounds["start"])
        end = pd.to_datetime(bounds["end"])
        labels.loc[dates.between(start, end, inclusive="both")] = name
    return labels


def _prepare_base_forward(forward_backtest_long: pd.DataFrame, spot_history_long: pd.DataFrame, vol_window_days: int) -> pd.DataFrame:
    required_forward = {"currency_pair", "side", "direction", "tenor_months", "transaction_date", "hedge_transaction_date", "hedge_n_days", "hedge_spot_rate", "forward_rate", "realized_spot", "realized_forward_advantage"}
    missing_forward = sorted(required_forward - set(forward_backtest_long.columns))
    if missing_forward:
        raise ValueError(f"Forward backtest table is missing required columns: {missing_forward}")
    required_history = {"date", "currency_pair", "spot_mid"}
    missing_history = sorted(required_history - set(spot_history_long.columns))
    if missing_history:
        raise ValueError(f"Spot history table is missing required columns: {missing_history}")

    base = forward_backtest_long.copy()
    base["transaction_date"] = pd.to_datetime(base["transaction_date"], errors="coerce")
    base["hedge_transaction_date"] = pd.to_datetime(base["hedge_transaction_date"], errors="coerce")
    base["hedge_n_days"] = pd.to_numeric(base["hedge_n_days"], errors="coerce")
    for col in ["hedge_spot_rate", "forward_rate", "realized_spot", "realized_forward_advantage"]:
        base[col] = pd.to_numeric(base[col], errors="coerce")
    base = base[base["hedge_transaction_date"].notna() & base["currency_pair"].notna()].copy()

    history = spot_history_long.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["spot_mid"] = pd.to_numeric(history["spot_mid"], errors="coerce")
    history = history[history["date"].notna() & history["spot_mid"].notna()].sort_values(["currency_pair", "date"]).copy()
    history["log_return"] = history.groupby("currency_pair", sort=False)["spot_mid"].transform(lambda s: np.log(s).diff())
    history["sigma_annual"] = history.groupby("currency_pair", sort=False)["log_return"].transform(
        lambda s: s.rolling(vol_window_days, min_periods=30).std(ddof=1) * np.sqrt(252.0)
    )
    sigma_table = history[["currency_pair", "date", "sigma_annual"]].drop_duplicates(["currency_pair", "date"]).sort_values(["currency_pair", "date"])
    sigma_table = sigma_table[sigma_table["date"].notna() & sigma_table["currency_pair"].notna()].copy()

    base = base.sort_values(["currency_pair", "hedge_transaction_date", "tenor_months", "transaction_date"]).reset_index(drop=True)
    base["currency_pair"] = base["currency_pair"].astype(str)
    sigma_table["currency_pair"] = sigma_table["currency_pair"].astype(str)
    base = base.sort_values(
        ["hedge_transaction_date", "currency_pair"], kind="mergesort"
    ).reset_index(drop=True)
    
    sigma_table = sigma_table.sort_values(
        ["date", "currency_pair"], kind="mergesort"
    ).reset_index(drop=True)
    merged = pd.merge_asof(
        base,
        sigma_table,
        left_on="hedge_transaction_date",
        right_on="date",
        by="currency_pair",
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.drop(columns=["date"], errors="ignore")
    merged["sigma_E"] = merged["sigma_annual"] * np.sqrt(pd.to_numeric(merged["hedge_n_days"], errors="coerce") / 360.0)
    merged = merged.drop(columns=["sigma_annual"])
    merged["regime_label"] = _assign_regime_label(merged["hedge_transaction_date"])
    return merged


def _vectorized_h_star(sigma_E: pd.Series, rho: float, sigma_Q: float, carry_cost: pd.Series, hedge_spot_rate: pd.Series, gamma_R: float) -> np.ndarray:
    sigma = pd.to_numeric(sigma_E, errors="coerce").to_numpy(dtype=float)
    carry = pd.to_numeric(carry_cost, errors="coerce").to_numpy(dtype=float)
    spot = pd.to_numeric(hedge_spot_rate, errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 2.0 * np.square(sigma) * spot
        gamma_term = np.where(denom == 0.0, 0.0, gamma_R * carry / denom)
        return 1.0 + (rho * sigma_Q) / sigma - gamma_term


def _build_rows_for_scenario(
    base: pd.DataFrame,
    hedge_scenario: str,
    target_intensity: float,
    stress_scenario: str,
    stress_bps: float,
    gamma_R: float | None,
    gamma_source: str,
    pop_median_rho: float,
    pop_median_sigma_Q: float,
) -> pd.DataFrame:
    df = base.copy()
    df["hedge_intensity_scenario"] = hedge_scenario
    df["target_intensity"] = float(target_intensity)
    df["forward_stress_scenario"] = stress_scenario
    df["carry_cost_used"] = pd.to_numeric(df["forward_rate"], errors="coerce") - pd.to_numeric(df["hedge_spot_rate"], errors="coerce")
    df["carry_cost_used"] = df["carry_cost_used"] + (float(stress_bps) / 10000.0) * pd.to_numeric(df["hedge_spot_rate"], errors="coerce") * pd.to_numeric(df["hedge_n_days"], errors="coerce") / 360.0

    active = hedge_scenario in {"low_protection", "baseline_protection", "high_protection"}
    if active and gamma_R is not None and np.isfinite(gamma_R):
        h_star = _vectorized_h_star(df["sigma_E"], pop_median_rho, pop_median_sigma_Q, df["carry_cost_used"], df["hedge_spot_rate"], float(gamma_R))
        h_c = np.clip(h_star, 0.0, 1.0)
        gamma_used = np.full(len(df), float(gamma_R), dtype=float)
        gamma_source_values = np.full(len(df), gamma_source, dtype=object)
    elif hedge_scenario == "no_hedge":
        h_star = np.zeros(len(df), dtype=float)
        h_c = np.zeros(len(df), dtype=float)
        gamma_used = np.full(len(df), np.nan, dtype=float)
        gamma_source_values = np.full(len(df), "rule_based_no_hedge", dtype=object)
    elif hedge_scenario == "full_hedge":
        h_star = np.ones(len(df), dtype=float)
        h_c = np.ones(len(df), dtype=float)
        gamma_used = np.full(len(df), np.nan, dtype=float)
        gamma_source_values = np.full(len(df), "rule_based_full_hedge", dtype=object)
    else:
        h_star = np.full(len(df), np.nan, dtype=float)
        h_c = np.full(len(df), np.nan, dtype=float)
        gamma_used = np.full(len(df), np.nan, dtype=float)
        gamma_source_values = np.full(len(df), gamma_source, dtype=object)

    if active and (gamma_R is None or not np.isfinite(gamma_R)):
        h_star = np.full(len(df), np.nan, dtype=float)
        h_c = np.full(len(df), np.nan, dtype=float)
        gamma_source_values = np.full(len(df), "split_calibration_failed_no_oos", dtype=object)

    extreme_mask = (~np.isfinite(h_star)) | (np.abs(h_star) > 100.0)
    market_status = np.where(~np.isfinite(df["sigma_E"]), "insufficient_vol_history", "ok")
    market_status = np.where(extreme_mask & (market_status == "ok"), "extreme_h_star", market_status)

    forward_advantage = pd.to_numeric(df["realized_forward_advantage"], errors="coerce").to_numpy(dtype=float)
    spot = pd.to_numeric(df["hedge_spot_rate"], errors="coerce").to_numpy(dtype=float)
    hedged_pnl = h_c * forward_advantage / spot
    unhedged_pnl = np.zeros(len(df), dtype=float)
    sigma = pd.to_numeric(df["sigma_E"], errors="coerce").to_numpy(dtype=float)
    var_unhedged = np.square(sigma) + pop_median_sigma_Q ** 2 + 2.0 * pop_median_rho * sigma * pop_median_sigma_Q
    var_hedged = np.square(1.0 - h_c) * np.square(sigma) + pop_median_sigma_Q ** 2 + 2.0 * (1.0 - h_c) * pop_median_rho * sigma * pop_median_sigma_Q
    var_unhedged = np.where(np.isfinite(var_unhedged), var_unhedged, np.nan)
    var_hedged = np.where(np.isfinite(var_hedged), var_hedged, np.nan)
    he_t = np.where(var_unhedged > 0, 1.0 - var_hedged / var_unhedged, np.nan)

    return pd.DataFrame({
        "run_id": df.get("run_id"),
        "currency_pair": df["currency_pair"],
        "side": df["side"],
        "direction": df["direction"],
        "tenor_months": df["tenor_months"],
        "transaction_date": df["transaction_date"],
        "hedge_transaction_date": df["hedge_transaction_date"],
        "hedge_n_days": df["hedge_n_days"],
        "regime_label": df["regime_label"],
        "hedge_intensity_scenario": df["hedge_intensity_scenario"],
        "target_intensity": df["target_intensity"],
        "forward_stress_scenario": df["forward_stress_scenario"],
        "hedge_spot_rate": df["hedge_spot_rate"],
        "forward_rate": df["forward_rate"],
        "realized_spot": df["realized_spot"],
        "carry_cost_used": df["carry_cost_used"],
        "forward_advantage": forward_advantage,
        "sigma_E": sigma,
        "pop_median_rho": float(pop_median_rho),
        "pop_median_sigma_Q": float(pop_median_sigma_Q),
        "gamma_R_used": gamma_used,
        "gamma_R_source": gamma_source_values,
        "h_star": h_star,
        "h_c": h_c,
        "hedged_pnl_per_unit": hedged_pnl,
        "unhedged_pnl_per_unit": unhedged_pnl,
        "relative_gain": hedged_pnl,
        "hit_indicator": (forward_advantage > 0).astype(int),
        "var_unhedged": var_unhedged,
        "var_hedged": var_hedged,
        "HE_t": he_t,
        "market_row_status": market_status,
    })


def compute_rolling_market_performance(
    forward_backtest_long: pd.DataFrame,
    spot_history_long: pd.DataFrame,
    accepted_profiles: pd.DataFrame,
    hedge_scenarios: dict,
    gamma_R_global: dict,
    stress_scenarios: dict,
    vol_window_days: int = 252,
    run_id: str | None = None,
) -> pd.DataFrame:
    base = _prepare_base_forward(forward_backtest_long, spot_history_long, vol_window_days)
    pop_median_rho = float(accepted_profiles["rho"].median()) if not accepted_profiles.empty else 0.0
    pop_median_sigma_Q = float(accepted_profiles["sigma_Q"].median()) if not accepted_profiles.empty else 0.0
    base = base.assign(rho_median=pop_median_rho, sigma_Q_median=pop_median_sigma_Q)

    frames: list[pd.DataFrame] = []
    for scenario, target_intensity in hedge_scenarios.items():
        gamma_R = float(gamma_R_global.get(scenario, np.nan)) if scenario in gamma_R_global else np.nan
        gamma_source = "global_calibration" if scenario not in {"no_hedge", "full_hedge"} else ("rule_based_no_hedge" if scenario == "no_hedge" else "rule_based_full_hedge")
        for stress_scenario, stress_bps in stress_scenarios.items():
            frame = _build_rows_for_scenario(base, scenario, target_intensity, stress_scenario, stress_bps, gamma_R, gamma_source, pop_median_rho, pop_median_sigma_Q)
            if run_id is not None:
                frame["run_id"] = run_id
            frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    cols = [
        "run_id", "currency_pair", "side", "direction", "tenor_months",
        "transaction_date", "hedge_transaction_date", "hedge_n_days", "regime_label",
        "hedge_intensity_scenario", "target_intensity", "forward_stress_scenario",
        "hedge_spot_rate", "forward_rate", "realized_spot", "carry_cost_used",
        "forward_advantage", "sigma_E", "pop_median_rho", "pop_median_sigma_Q",
        "gamma_R_used", "gamma_R_source", "h_star", "h_c", "hedged_pnl_per_unit",
        "unhedged_pnl_per_unit", "relative_gain", "hit_indicator", "var_unhedged",
        "var_hedged", "HE_t", "market_row_status",
    ]
    return result[cols].reset_index(drop=True)


def _calibrate_split_gamma(benchmark: pd.DataFrame, target_intensity: float) -> tuple[float, str, int]:
    if benchmark.empty:
        return np.nan, "calibration_failed", 0
    med_sigma = float(pd.to_numeric(benchmark["sigma_E"], errors="coerce").median())
    med_s0 = float(pd.to_numeric(benchmark["hedge_spot_rate"], errors="coerce").median())
    med_carry = float(pd.to_numeric(benchmark["carry_cost_base"], errors="coerce").median())
    rho = float(pd.to_numeric(benchmark["rho_median"], errors="coerce").median())
    sigma_q = float(pd.to_numeric(benchmark["sigma_Q_median"], errors="coerce").median())
    try:
        gamma = solve_gamma_for_target(target_intensity, 1.0, med_sigma, rho, sigma_q, med_s0, med_carry)
        return float(gamma), "ok", int(len(benchmark))
    except Exception:
        return np.nan, "calibration_failed", int(len(benchmark))


def compute_oos_market_performance(
    forward_backtest_long: pd.DataFrame,
    spot_history_long: pd.DataFrame,
    accepted_profiles: pd.DataFrame,
    hedge_scenarios: dict,
    stress_scenarios: dict,
    vol_window_days: int = 252,
    walk_forward_initial_train_start: str = "2008-01-01",
    walk_forward_initial_train_end: str = "2013-12-31",
    run_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = _prepare_base_forward(forward_backtest_long, spot_history_long, vol_window_days)
    pop_median_rho = float(accepted_profiles["rho"].median()) if not accepted_profiles.empty else 0.0
    pop_median_sigma_Q = float(accepted_profiles["sigma_Q"].median()) if not accepted_profiles.empty else 0.0
    base = base.assign(rho_median=pop_median_rho, sigma_Q_median=pop_median_sigma_Q)
    train_start = pd.to_datetime(walk_forward_initial_train_start)
    benchmark = base[(base["currency_pair"] == "EUR_TND") & (base["tenor_months"] == 6)].copy()
    benchmark["carry_cost_base"] = pd.to_numeric(benchmark["forward_rate"], errors="coerce") - pd.to_numeric(benchmark["hedge_spot_rate"], errors="coerce")

    calibration_rows = []
    performance_frames = []
    active_scenarios = [k for k in hedge_scenarios.keys() if k not in {"no_hedge", "full_hedge"}]
    for split_index, test_year in enumerate(range(2014, 2026), start=1):
        split_id = f"wf_{split_index:03d}"
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")
        current_train_end = pd.Timestamp(f"{test_year - 1}-12-31")
        train_rows = benchmark[(benchmark["hedge_transaction_date"] >= train_start) & (benchmark["hedge_transaction_date"] <= current_train_end)].copy()
        for scenario in active_scenarios:
            target = float(hedge_scenarios[scenario])
            gamma, status, n_train = _calibrate_split_gamma(train_rows, target)
            calibration_rows.append({
                "split_id": split_id,
                "train_start": train_start,
                "train_end": current_train_end,
                "test_start": test_start,
                "test_end": test_end,
                "hedge_intensity_scenario": scenario,
                "target_intensity": target,
                "gamma_R": gamma,
                "calibration_status": status,
                "n_training_observations": n_train,
                "calibration_currency_pair": "EUR_TND",
                "calibration_tenor_months": 6,
            })

        split_base = base[(base["hedge_transaction_date"] >= test_start) & (base["hedge_transaction_date"] <= test_end)].copy()
        if split_base.empty:
            continue

        gamma_map = {row["hedge_intensity_scenario"]: row["gamma_R"] for row in calibration_rows if row["split_id"] == split_id}
        gamma_status = {row["hedge_intensity_scenario"]: row["calibration_status"] for row in calibration_rows if row["split_id"] == split_id}

        for scenario, target_intensity in hedge_scenarios.items():
            if scenario in {"no_hedge", "full_hedge"}:
                gamma_R = np.nan
                gamma_source = "rule_based_no_hedge" if scenario == "no_hedge" else "rule_based_full_hedge"
                for stress_scenario, stress_bps in stress_scenarios.items():
                    frame = _build_rows_for_scenario(split_base, scenario, target_intensity, stress_scenario, stress_bps, gamma_R, gamma_source, pop_median_rho, pop_median_sigma_Q)
                    frame["split_specific_gamma_used"] = True
                    frame["methodological_status"] = "true_out_of_sample"
                    frame["split_id"] = split_id
                    frame["train_start"] = train_start
                    frame["train_end"] = current_train_end
                    frame["test_start"] = test_start
                    frame["test_end"] = test_end
                    performance_frames.append(frame)
                continue

            gamma_R = gamma_map.get(scenario, np.nan)
            status = gamma_status.get(scenario, "calibration_failed")
            gamma_source = "split_training_calibration" if np.isfinite(gamma_R) and status == "ok" else "split_calibration_failed_no_oos"
            for stress_scenario, stress_bps in stress_scenarios.items():
                frame = _build_rows_for_scenario(split_base, scenario, target_intensity, stress_scenario, stress_bps, gamma_R, gamma_source, pop_median_rho, pop_median_sigma_Q)
                frame["split_specific_gamma_used"] = True
                frame["methodological_status"] = "true_out_of_sample" if np.isfinite(gamma_R) and status == "ok" else "calibration_failed"
                if not np.isfinite(gamma_R) or status != "ok":
                    frame["h_c"] = np.nan
                frame["split_id"] = split_id
                frame["train_start"] = train_start
                frame["train_end"] = current_train_end
                frame["test_start"] = test_start
                frame["test_end"] = test_end
                performance_frames.append(frame)

    performance = pd.concat(performance_frames, ignore_index=True) if performance_frames else pd.DataFrame()
    calibration = pd.DataFrame(calibration_rows)
    if not performance.empty and run_id is not None:
        performance["run_id"] = run_id
    if not calibration.empty and run_id is not None:
        calibration["run_id"] = run_id
    return performance, calibration
