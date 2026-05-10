from __future__ import annotations

import numpy as np
import pandas as pd

from .gamma_calibration import calibrate_gamma_R
from .hedge_engine import compute_hedge_decisions


def row_pnl(row: dict) -> float:
    h_c = float(row["h_c"])
    if row["direction"] == "outflow":
        realized = float(row["realized_future_spot_ask"])
        return h_c * (float(row["F_executable"]) - realized) - (1.0 - h_c) * (realized - float(row["S0"]))
    realized = float(row["realized_future_spot_bid"])
    return h_c * (realized - float(row["F_executable"])) + (1.0 - h_c) * (realized - float(row["S0"]))


def assign_sub_period(date_value, sub_periods: dict) -> str | None:
    if pd.isna(date_value):
        return None
    date = pd.to_datetime(date_value)
    for name, bounds in sub_periods.items():
        if pd.to_datetime(bounds["start"]) <= date <= pd.to_datetime(bounds["end"]):
            return name
    return "out_of_sample"


def run_backtest(decisions: pd.DataFrame, sub_periods: dict, require_backtest: bool = False) -> pd.DataFrame:
    if decisions.empty:
        if require_backtest:
            raise ValueError("Backtest required but Stage_2_Decisions is empty.")
        return pd.DataFrame([{
            "status": "backtest_skipped_missing_realized_future_spot",
            "n_rows_missing_realized_future_spot": 0,
            "n_rows_backtested": 0,
            "missing_realized_spot_warning": False,
            "n_observations": 0,
        }])
    if "direction" not in decisions.columns:
        if require_backtest:
            raise ValueError("Backtest required but decisions are missing direction column.")
        return pd.DataFrame([{
            "status": "backtest_skipped_missing_realized_future_spot",
            "n_rows_missing_realized_future_spot": int(len(decisions)),
            "n_rows_backtested": 0,
            "missing_realized_spot_warning": True,
            "n_observations": 0,
        }])
    df = decisions.copy()
    df["realized_future_spot"] = np.where(
        df["direction"] == "outflow",
        df.get("realized_future_spot_ask"),
        df.get("realized_future_spot_bid"),
    )
    df["realized_spot_status"] = np.where(df["realized_future_spot"].notna(), "ok", "missing_realized_future_spot")
    missing_count = int((df["realized_spot_status"] != "ok").sum())
    if missing_count > 0 and require_backtest:
        raise ValueError(
            f"Backtest required but {missing_count} rows are missing realized future spot values "
            f"(direction-aware bid/ask requirement)."
        )
    df = df[df["realized_spot_status"] == "ok"].copy()
    if df.empty:
        return pd.DataFrame([{
            "status": "backtest_skipped_missing_realized_future_spot",
            "n_rows_missing_realized_future_spot": missing_count,
            "n_rows_backtested": 0,
            "missing_realized_spot_warning": missing_count > 0,
            "n_observations": 0,
        }])
    df["pnl"] = [row_pnl(r) for r in df.to_dict("records")]
    df["pnl_normalized"] = df["pnl"] / df["S0"]
    df["sub_period"] = df["realized_future_date"].apply(lambda d: assign_sub_period(d, sub_periods))
    keys = ["profile_id", "currency_pair", "timing_cv_scenario", "hedge_intensity_scenario", "tenor_months", "sub_period"]
    rows = []
    for key, grp in df.groupby(keys, dropna=False):
        vals = grp["pnl_normalized"].sort_values()
        std = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
        cutoff = max(1, int(np.ceil(0.05 * len(vals))))
        rows.append(dict(zip(keys, key), **{
            "mean_pnl": float(vals.mean()), "std_pnl": std,
            "sharpe_like": float(vals.mean() / std) if std and not pd.isna(std) and std > 0 else np.nan,
            "hit_rate": float((vals > 0).mean()), "cvar_5": float(vals.iloc[:cutoff].mean()),
            "worst_case": float(vals.min()), "n_observations": int(len(vals)),
            "n_rows_missing_realized_future_spot": missing_count,
            "n_rows_backtested": int(len(df)),
            "missing_realized_spot_warning": bool(missing_count > 0),
            "status": "backtest_completed" if missing_count == 0 else "backtest_completed_with_missing_rows_dropped",
        }))
    return pd.DataFrame(rows)


def _market_snapshot_from_training(decisions_train: pd.DataFrame) -> pd.DataFrame:
    if decisions_train.empty:
        return pd.DataFrame(columns=["currency_pair", "tenor_months", "sigma_E", "carry_cost", "spot_mid"])
    return decisions_train.groupby(["currency_pair", "tenor_months"], as_index=False).agg(
        sigma_E=("sigma_E", "median"),
        carry_cost=("carry_cost", "median"),
        spot_mid=("S0", "median"),
    )


def gamma_calibration_by_split(
    splits: pd.DataFrame,
    decisions: pd.DataFrame,
    accepted: pd.DataFrame,
    diagnostics: pd.DataFrame,
    hedge_scenarios: dict[str, float],
) -> pd.DataFrame:
    rows = []
    if splits.empty:
        return pd.DataFrame(columns=[
            "split_id", "split_type", "regime_name", "regime_label", "train_start", "train_end",
            "test_start", "test_end", "hedge_intensity_scenario", "target_intensity", "gamma_R",
            "calibration_currency_pair", "calibration_tenor_months", "calibration_status", "n_training_observations",
        ])
    d = decisions.copy()
    d["valuation_date"] = pd.to_datetime(d["valuation_date"], errors="coerce")
    global_market = _market_snapshot_from_training(d)
    for split in splits.to_dict("records"):
        train_start = pd.to_datetime(split["train_start"])
        train_end = pd.to_datetime(split["train_end"])
        train = d[(d["valuation_date"] >= train_start) & (d["valuation_date"] <= train_end)]
        n_train = int(len(train))
        market_train = _market_snapshot_from_training(train)
        if market_train.empty:
            if global_market.empty:
                for scen, tau in hedge_scenarios.items():
                    rows.append({
                        "split_id": split["split_id"], "split_type": split["split_type"], "regime_name": split["regime_name"],
                        "regime_label": split["regime_label"], "train_start": train_start, "train_end": train_end,
                        "test_start": pd.to_datetime(split["test_start"]), "test_end": pd.to_datetime(split["test_end"]),
                        "hedge_intensity_scenario": scen, "target_intensity": tau, "gamma_R": np.nan,
                        "calibration_currency_pair": None, "calibration_tenor_months": None,
                        "calibration_status": "insufficient_training_data", "n_training_observations": n_train,
                    })
                continue
            market_train = global_market
            status_note = "ok_proxy_single_valuation_snapshot"
        else:
            status_note = "ok"
        gamma_map, _ = calibrate_gamma_R(accepted, diagnostics, market_train, hedge_scenarios)
        for scen, tau in hedge_scenarios.items():
            gamma = np.nan if scen in {"no_hedge", "full_hedge"} else float(gamma_map.get(scen, np.nan))
            rows.append({
                "split_id": split["split_id"], "split_type": split["split_type"], "regime_name": split["regime_name"],
                "regime_label": split["regime_label"], "train_start": train_start, "train_end": train_end,
                "test_start": pd.to_datetime(split["test_start"]), "test_end": pd.to_datetime(split["test_end"]),
                "hedge_intensity_scenario": scen, "target_intensity": tau, "gamma_R": gamma,
                "calibration_currency_pair": "EUR_TND", "calibration_tenor_months": 6,
                "calibration_status": status_note if np.isfinite(gamma) or scen in {"no_hedge", "full_hedge"} else "fallback_or_nan",
                "n_training_observations": n_train,
            })
    return pd.DataFrame(rows)


def _variance_reduction(series_unhedged: pd.Series, series_hedged: pd.Series) -> float:
    u = float(series_unhedged.mean()) if len(series_unhedged) else np.nan
    h = float(series_hedged.mean()) if len(series_hedged) else np.nan
    if pd.isna(u) or u <= 0 or pd.isna(h):
        return np.nan
    return 1.0 - h / u


def _robustness_comment(n_obs: int, mean_pnl: float, hit_rate: float, mean_he: float, worst_case: float) -> str:
    if n_obs < 30:
        return "insufficient_data"
    if pd.notna(mean_he) and mean_he < -0.2:
        return "unstable_regime"
    if pd.notna(mean_pnl) and mean_pnl > 0 and pd.notna(hit_rate) and hit_rate >= 0.5 and pd.notna(mean_he) and mean_he >= 0:
        return "robust_positive"
    if pd.notna(mean_pnl) and mean_pnl < 0 and pd.notna(mean_he) and mean_he < 0:
        return "hedge_costly"
    return "mixed"


def run_split_backtests(
    handoff: pd.DataFrame,
    market_snapshot: pd.DataFrame,
    valuation_date: str,
    hedge_scenarios: dict[str, float],
    gamma_global: dict[str, float],
    gamma_by_split: pd.DataFrame,
    splits: pd.DataFrame,
    require_realized_future_spot: bool = False,
    split_workflow_label: str = "walk_forward",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if handoff.empty or market_snapshot.empty or splits.empty:
        status = "skipped_missing_realized_future_spot"
        status_row = pd.DataFrame([{
            "status": status,
            "backtest_method": "split_specific_recompute",
            "split_specific_gamma_used": False,
            "methodological_status": "diagnostic_not_true_out_of_sample",
            "split_workflow": split_workflow_label,
            "n_rows_missing_realized_future_spot": 0,
            "n_rows_backtested": 0,
        }])
        return status_row.copy(), status_row.copy(), status_row.copy(), status_row.copy(), status

    calibration = gamma_by_split.copy() if isinstance(gamma_by_split, pd.DataFrame) else pd.DataFrame()
    calibration["split_id"] = calibration.get("split_id", pd.Series(dtype=str)).astype(str)
    oos_rows = []
    total_missing = 0
    total_backtested = 0
    for split in splits.to_dict("records"):
        split_id = str(split.get("split_id", ""))
        split_cal = calibration[calibration["split_id"] == split_id] if not calibration.empty else pd.DataFrame()
        gamma_override = {}
        valid_gamma = False
        for scen in hedge_scenarios:
            if scen in {"no_hedge", "full_hedge"}:
                continue
            val = np.nan
            if not split_cal.empty:
                rows = split_cal[split_cal["hedge_intensity_scenario"] == scen]
                if not rows.empty:
                    val = pd.to_numeric(rows.iloc[0].get("gamma_R"), errors="coerce")
                    status = str(rows.iloc[0].get("calibration_status", ""))
                    if status.startswith("ok") and np.isfinite(val):
                        gamma_override[scen] = float(val)
                        valid_gamma = True
        # Recompute split decisions using split-specific gamma when available.
        gamma_source_label = f"split_{split_id}" if valid_gamma else "global_calibration"
        recomputed = compute_hedge_decisions(
            handoff,
            market_snapshot,
            valuation_date,
            hedge_scenarios,
            gamma_global,
            gamma_override_by_scenario=gamma_override if valid_gamma else None,
            gamma_source_label=gamma_source_label,
        )
        recomputed["valuation_date"] = pd.to_datetime(recomputed["valuation_date"], errors="coerce")
        test_start = pd.to_datetime(split["test_start"])
        test_end = pd.to_datetime(split["test_end"])
        test = recomputed[(recomputed["valuation_date"] >= test_start) & (recomputed["valuation_date"] <= test_end)].copy()
        if test.empty:
            continue
        test["realized_future_spot"] = np.where(
            test["direction"] == "outflow",
            test.get("realized_future_spot_ask"),
            test.get("realized_future_spot_bid"),
        )
        test["realized_spot_status"] = np.where(test["realized_future_spot"].notna(), "ok", "missing_realized_future_spot")
        missing_count = int((test["realized_spot_status"] != "ok").sum())
        if missing_count > 0 and require_realized_future_spot:
            raise ValueError(
                f"realized future spot data are required for split {split_id} but {missing_count} direction-aware rows are missing."
            )
        test = test[test["realized_spot_status"] == "ok"].copy()
        if test.empty:
            continue
        test["split_id"] = split["split_id"]
        test["split_type"] = split["split_type"]
        test["regime_name"] = split["regime_name"]
        test["regime_label"] = split["regime_label"]
        test["train_start"] = pd.to_datetime(split["train_start"])
        test["train_end"] = pd.to_datetime(split["train_end"])
        test["test_start"] = test_start
        test["test_end"] = test_end
        test["pnl"] = [row_pnl(r) for r in test.to_dict("records")]
        test["pnl_normalized"] = test["pnl"] / test["S0"]
        test["backtest_method"] = "split_specific_recompute" if valid_gamma else "global_decisions_filtered"
        test["split_specific_gamma_used"] = bool(valid_gamma)
        test["methodological_status"] = "true_out_of_sample" if valid_gamma else "diagnostic_not_true_out_of_sample"
        test["split_workflow"] = split_workflow_label
        test["n_rows_missing_realized_future_spot"] = missing_count
        test["n_rows_backtested"] = int(len(test))
        test["backtest_row_status"] = "ok"
        test["gamma_R_used"] = test.get("gamma_R_used", test.get("gamma_R"))
        if not valid_gamma:
            test["gamma_R_source"] = "split_calibration_unavailable"
        oos_rows.append(test)
        total_missing += missing_count
        total_backtested += int(len(test))
    if not oos_rows:
        status = "skipped_missing_realized_future_spot"
        status_row = pd.DataFrame([{
            "status": status,
            "n_rows_missing_realized_future_spot": total_missing,
            "n_rows_backtested": total_backtested,
            "backtest_method": "split_specific_recompute",
            "split_specific_gamma_used": False,
            "methodological_status": "diagnostic_not_true_out_of_sample",
            "split_workflow": split_workflow_label,
        }])
        return status_row.copy(), status_row.copy(), status_row.copy(), status_row.copy(), status
    oos = pd.concat(oos_rows, ignore_index=True)
    oos_cols = [
        "split_id", "split_type", "regime_name", "regime_label", "train_start", "train_end",
        "test_start", "test_end", "profile_id", "family", "currency_pair", "tenor_months",
        "timing_cv_scenario", "hedge_intensity_scenario", "valuation_date", "realized_future_date",
        "E_t", "direction", "lambda", "h_star", "h_c", "gamma_R_used", "F_executable", "realized_future_spot",
        "pnl", "pnl_normalized", "HE_t", "expected_cost", "variance_unhedged", "variance_hedged",
        "backtest_method", "split_specific_gamma_used", "methodological_status", "split_workflow",
        "n_rows_missing_realized_future_spot", "n_rows_backtested", "backtest_row_status", "realized_spot_status",
    ]
    oos = oos[oos_cols]

    split_summary = oos.groupby(
        ["split_id", "split_type", "regime_name", "regime_label", "test_start", "test_end", "hedge_intensity_scenario", "family", "currency_pair", "tenor_months"],
        as_index=False,
    ).agg(
        n_observations=("pnl_normalized", "size"),
        mean_pnl=("pnl_normalized", "mean"),
        median_pnl=("pnl_normalized", "median"),
        std_pnl=("pnl_normalized", "std"),
        hit_rate=("pnl_normalized", lambda s: float((s > 0).mean())),
        worst_case=("pnl_normalized", "min"),
        cvar_5=("pnl_normalized", lambda s: float(s.nsmallest(max(1, int(np.ceil(0.05 * len(s))))).mean())),
        mean_h_c=("h_c", "mean"),
        gamma_R_used=("gamma_R_used", "median"),
        mean_expected_cost=("expected_cost", "mean"),
        mean_HE_t=("HE_t", "mean"),
        pct_positive_HE=("HE_t", lambda s: float((s > 0).mean())),
        pct_negative_HE=("HE_t", lambda s: float((s < 0).mean())),
        split_specific_gamma_used=("split_specific_gamma_used", "max"),
    )
    split_summary["backtest_method"] = "split_specific_recompute"
    split_summary["methodological_status"] = np.where(
        split_summary["split_specific_gamma_used"].astype(bool),
        "true_out_of_sample",
        "diagnostic_not_true_out_of_sample",
    )
    split_summary["split_workflow"] = split_workflow_label
    split_summary["n_rows_missing_realized_future_spot"] = total_missing
    split_summary["n_rows_backtested"] = int(total_backtested)
    var_red = oos.groupby(
        ["split_id", "split_type", "regime_name", "regime_label", "test_start", "test_end", "hedge_intensity_scenario", "family", "currency_pair", "tenor_months"]
    ).apply(lambda g: _variance_reduction(g["variance_unhedged"], g["variance_hedged"]))
    split_summary["variance_reduction"] = split_summary.set_index(
        ["split_id", "split_type", "regime_name", "regime_label", "test_start", "test_end", "hedge_intensity_scenario", "family", "currency_pair", "tenor_months"]
    ).index.map(var_red.to_dict())

    regime = split_summary.copy()
    regime_agg = regime.groupby(["regime_name", "regime_label", "split_id", "test_start", "test_end", "hedge_intensity_scenario"], as_index=False).agg(
        mean_pnl=("mean_pnl", "mean"),
        hit_rate=("hit_rate", "mean"),
        mean_HE_t=("mean_HE_t", "mean"),
        variance_reduction=("variance_reduction", "mean"),
        worst_case=("worst_case", "min"),
        n_observations=("n_observations", "sum"),
        gamma_R_used=("gamma_R_used", "median"),
        split_specific_gamma_used=("split_specific_gamma_used", "max"),
    )
    regime_agg["comment"] = regime_agg.apply(
        lambda r: _robustness_comment(int(r["n_observations"]), float(r["mean_pnl"]), float(r["hit_rate"]), float(r["mean_HE_t"]), float(r["worst_case"])),
        axis=1,
    )
    regime_agg["backtest_method"] = "split_specific_recompute"
    regime_agg["methodological_status"] = np.where(
        regime_agg["split_specific_gamma_used"].astype(bool),
        "true_out_of_sample",
        "diagnostic_not_true_out_of_sample",
    )
    regime_agg["split_workflow"] = split_workflow_label
    regime_agg["n_rows_missing_realized_future_spot"] = total_missing
    regime_agg["n_rows_backtested"] = int(total_backtested)
    robustness = regime_agg[["regime_name", "hedge_intensity_scenario", "mean_pnl", "hit_rate", "mean_HE_t", "variance_reduction", "worst_case", "comment", "gamma_R_used"]]
    return oos, split_summary, split_summary, regime_agg, "completed"
