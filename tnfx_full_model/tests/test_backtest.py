import pandas as pd

from tnfx_full_model.backtest import assign_sub_period, row_pnl, run_backtest, run_split_backtests
from tnfx_full_model.forward_pricing import price_market_snapshot
from tnfx_full_model.train_test_split import build_full_walk_forward_splits


def test_pnl_formulas_and_subperiod():
    out = dict(direction="outflow", h_c=0.5, F_executable=3.4, realized_future_spot_ask=3.5, S0=3.3)
    assert row_pnl(out) == 0.5 * (3.4 - 3.5) - 0.5 * (3.5 - 3.3)
    inc = dict(direction="inflow", h_c=0.5, F_executable=3.4, realized_future_spot_bid=3.5, S0=3.3)
    assert row_pnl(inc) == 0.5 * (3.5 - 3.4) + 0.5 * (3.5 - 3.3)
    periods = {"recent_2019_2025": {"start": "2019-01-01", "end": "2025-12-31"}}
    assert assign_sub_period("2024-01-01", periods) == "recent_2019_2025"
    df = pd.DataFrame([{**out, "profile_id": 1, "currency_pair": "EUR_TND", "timing_cv_scenario": "baseline",
                        "hedge_intensity_scenario": "baseline", "tenor_months": 6, "realized_future_spot_bid": 3.48,
                        "realized_future_date": "2024-01-01"}])
    res = run_backtest(df, periods)
    assert res["n_observations"].iloc[0] == 1


def test_backtest_skips_when_realized_missing():
    df = pd.DataFrame([{"realized_future_spot_bid": None, "realized_future_spot_ask": None}])
    res = run_backtest(df, {})
    assert res["status"].iloc[0] == "backtest_skipped_missing_realized_future_spot"


def test_backtest_partial_missing_realized_spot_drops_only_missing_rows():
    periods = {"recent_2019_2025": {"start": "2019-01-01", "end": "2025-12-31"}}
    df = pd.DataFrame([
        {
            "profile_id": 1, "currency_pair": "EUR_TND", "timing_cv_scenario": "baseline",
            "hedge_intensity_scenario": "baseline_protection", "tenor_months": 6,
            "direction": "outflow", "h_c": 0.5, "F_executable": 3.4, "S0": 3.3,
            "realized_future_spot_bid": 3.45, "realized_future_spot_ask": 3.5, "realized_future_date": "2024-01-01",
        },
        {
            "profile_id": 2, "currency_pair": "USD_TND", "timing_cv_scenario": "baseline",
            "hedge_intensity_scenario": "baseline_protection", "tenor_months": 6,
            "direction": "outflow", "h_c": 0.5, "F_executable": 3.4, "S0": 3.3,
            "realized_future_spot_bid": 3.45, "realized_future_spot_ask": None, "realized_future_date": "2024-01-01",
        },
    ])
    res = run_backtest(df, periods, require_backtest=False)
    assert set(res["status"].unique()).issubset({"backtest_completed_with_missing_rows_dropped", "backtest_completed"})
    assert int(res["n_rows_missing_realized_future_spot"].max()) == 1
    assert int(res["n_rows_backtested"].max()) == 1


def test_backtest_all_missing_returns_skipped_when_not_required():
    df = pd.DataFrame([
        {"direction": "outflow", "realized_future_spot_bid": 3.2, "realized_future_spot_ask": None},
        {"direction": "inflow", "realized_future_spot_bid": None, "realized_future_spot_ask": 3.3},
    ])
    res = run_backtest(df, {}, require_backtest=False)
    assert res["status"].iloc[0] == "backtest_skipped_missing_realized_future_spot"
    assert int(res["n_rows_backtested"].iloc[0]) == 0


def test_backtest_missing_realized_spot_raises_when_required():
    df = pd.DataFrame([{
        "profile_id": 1, "currency_pair": "EUR_TND", "timing_cv_scenario": "baseline",
        "hedge_intensity_scenario": "baseline", "tenor_months": 6, "direction": "outflow",
        "h_c": 0.4, "F_executable": 3.4, "S0": 3.3,
        "realized_future_spot_bid": 3.4, "realized_future_spot_ask": None, "realized_future_date": "2024-01-01",
    }])
    try:
        run_backtest(df, {}, require_backtest=True)
    except ValueError as exc:
        assert "missing realized future spot" in str(exc).lower()
        return
    raise AssertionError("Expected ValueError when require_backtest=True and realized spot is missing.")


def test_first_walk_forward_split_matches_config():
    dates = pd.date_range("2008-01-01", "2025-12-31", freq="365D")
    market = pd.DataFrame({"valuation_date": dates, "currency_pair": ["EUR_TND"] * len(dates)})
    cfg = {
        "backtest_split": {
            "walk_forward_initial_train_start": "2008-01-01",
            "walk_forward_initial_train_end": "2013-12-31",
            "walk_forward_test_frequency": "yearly",
            "expanding_window": True,
        }
    }
    splits = build_full_walk_forward_splits(market, cfg)
    first = splits.iloc[0]
    assert str(pd.to_datetime(first["train_start"]).date()) == "2008-01-01"
    assert str(pd.to_datetime(first["train_end"]).date()) == "2013-12-31"
    assert str(pd.to_datetime(first["test_start"]).date()) == "2014-01-01"
    assert str(pd.to_datetime(first["test_end"]).date()) == "2014-12-31"


def _base_handoff_market():
    handoff = pd.DataFrame([{
        "profile_id": 1, "family": "importer", "currency": "EUR", "currency_pair": "EUR_TND",
        "timing_cv_scenario": "baseline", "tenor_months": 6, "E_t": -0.12, "direction": "outflow",
        "delta_net_k": -0.12, "lambda": 0.9, "rho": 0.1, "sigma_Q": 0.2,
    }])
    market = pd.DataFrame([{
        "valuation_date": "2024-01-01", "currency_pair": "EUR_TND", "tenor_months": 6, "tenor_days": 180,
        "spot_bid": 3.30, "spot_ask": 3.32, "spot_mid": 3.31,
        "tnd_rate_bid": 0.07, "tnd_rate_ask": 0.08, "fcy_rate_bid": 0.02, "fcy_rate_ask": 0.03,
        "forward_bid": float("nan"), "forward_ask": float("nan"),
        "realized_future_date": "2024-07-01", "realized_future_spot_bid": 3.35, "realized_future_spot_ask": 3.37,
    }])
    vol = pd.DataFrame([{"currency_pair": "EUR_TND", "tenor_months": 6, "tenor_days": 180, "sigma_E": 0.15}])
    snapshot = price_market_snapshot(market, vol)
    splits = pd.DataFrame([
        {"split_id": "wf_001", "split_type": "full_walk_forward", "regime_name": "wf", "regime_label": "wf",
         "train_start": "2008-01-01", "train_end": "2013-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
        {"split_id": "wf_002", "split_type": "full_walk_forward", "regime_name": "wf", "regime_label": "wf",
         "train_start": "2008-01-01", "train_end": "2014-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
    ])
    gamma_table = pd.DataFrame([
        {"split_id": "wf_001", "hedge_intensity_scenario": "baseline_protection", "gamma_R": 0.1, "calibration_status": "ok"},
        {"split_id": "wf_002", "hedge_intensity_scenario": "baseline_protection", "gamma_R": 9.0, "calibration_status": "ok"},
    ])
    for sid in ["wf_001", "wf_002"]:
        for scen in ["no_hedge", "low_protection", "high_protection", "full_hedge"]:
            gamma_table = pd.concat([gamma_table, pd.DataFrame([{
                "split_id": sid, "hedge_intensity_scenario": scen, "gamma_R": float("nan"), "calibration_status": "ok",
            }])], ignore_index=True)
    return handoff, snapshot, splits, gamma_table


def test_true_oos_uses_split_specific_gamma():
    handoff, snapshot, splits, gamma_table = _base_handoff_market()
    oos, _, _, _, _ = run_split_backtests(
        handoff, snapshot, "2024-01-01",
        {"no_hedge": 0.0, "baseline_protection": 0.5, "full_hedge": 1.0},
        {"baseline_protection": 1.0},
        gamma_table, splits, require_realized_future_spot=False, split_workflow_label="walk_forward",
    )
    rows = oos[oos["hedge_intensity_scenario"] == "baseline_protection"].copy()
    assert not rows.empty
    assert rows["split_specific_gamma_used"].all()
    hc_by_split = rows.groupby("split_id")["h_c"].mean().to_dict()
    assert len(set(round(v, 10) for v in hc_by_split.values())) > 1


def test_split_with_nan_gamma_marks_diagnostic_not_silently_global():
    handoff, snapshot, splits, gamma_table = _base_handoff_market()
    gamma_table.loc[gamma_table["hedge_intensity_scenario"] == "baseline_protection", "gamma_R"] = float("nan")
    gamma_table.loc[gamma_table["hedge_intensity_scenario"] == "baseline_protection", "calibration_status"] = "insufficient_training_data"
    oos, _, _, _, _ = run_split_backtests(
        handoff, snapshot, "2024-01-01",
        {"no_hedge": 0.0, "baseline_protection": 0.5, "full_hedge": 1.0},
        {"baseline_protection": 1.0},
        gamma_table, splits, require_realized_future_spot=False, split_workflow_label="walk_forward",
    )
    assert not oos.empty
    assert (oos["split_specific_gamma_used"] == False).all()
    assert (oos["methodological_status"] == "diagnostic_not_true_out_of_sample").all()
