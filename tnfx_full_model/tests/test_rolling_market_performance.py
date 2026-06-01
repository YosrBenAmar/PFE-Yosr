import numpy as np
import pandas as pd

from tnfx_full_model.rolling_market_performance import compute_rolling_market_performance


def _sample_inputs():
    forward = pd.DataFrame([
        {
            "run_id": "x", "source_sheet": "EUR ASK Forward", "currency_pair": "EUR_TND", "currency": "EUR",
            "side": "ASK", "direction": "outflow", "tenor_months": 6, "transaction_type": "Export",
            "transaction_date": "2024-06-30", "hedge_transaction_date": "2024-01-15", "hedge_n_days": 180,
            "hedge_spot_rate": 3.30, "domestic_yield": 0.08, "foreign_yield": 0.03, "forward_rate": 3.38,
            "realized_spot": 3.25, "realized_forward_advantage": -0.13, "F_recomputed": 3.38,
            "cip_recalculation_error": 0.0, "cip_recalculation_status": "ok",
        }
        for _ in range(10)
    ])
    history = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=300, freq="D"),
        "currency_pair": ["EUR_TND"] * 300,
        "spot_mid": np.linspace(3.0, 3.4, 300),
    })
    accepted = pd.DataFrame({"family": ["importer", "exporter"], "rho": [0.1, 0.2], "sigma_Q": [0.2, 0.3]})
    handoff = pd.DataFrame({
        "profile_id": [1, 2, 3],
        "currency_pair": ["EUR_TND"] * 3,
        "tenor_months": [6, 6, 6],
        "direction": ["outflow"] * 3,
        "E_t": [-0.10, -0.15, -0.12],
    })
    return forward, history, accepted, handoff


def test_rolling_market_performance_row_count():
    forward, history, accepted, handoff = _sample_inputs()
    res = compute_rolling_market_performance(
        forward_backtest_long=forward,
        spot_history_long=history,
        accepted_profiles=accepted,
        stage_1_5_handoff=handoff,
        hedge_scenarios={"no_hedge": 0.0, "low_protection": 0.25, "baseline_protection": 0.5, "high_protection": 0.75, "full_hedge": 1.0},
        gamma_R_global={"low_protection": 0.5, "baseline_protection": 0.5, "high_protection": 0.5},
        stress_scenarios={"cip_base": 0, "cip_plus_50bps": 50},
        vol_window_days=30,
        run_id="x",
    )
    assert len(res) == 10 * 5 * 2


def test_rolling_market_performance_no_lookahead():
    forward = pd.DataFrame([{ 
        "run_id": "x", "source_sheet": "EUR ASK Forward", "currency_pair": "EUR_TND", "currency": "EUR",
        "side": "ASK", "direction": "outflow", "tenor_months": 6, "transaction_type": "Export",
        "transaction_date": "2024-06-30", "hedge_transaction_date": "2024-02-10", "hedge_n_days": 180,
        "hedge_spot_rate": 3.30, "domestic_yield": 0.08, "foreign_yield": 0.03, "forward_rate": 3.38,
        "realized_spot": 3.25, "realized_forward_advantage": -0.13, "F_recomputed": 3.38,
        "cip_recalculation_error": 0.0, "cip_recalculation_status": "ok",
    }])
    pre_dates = pd.date_range("2023-01-01", periods=35, freq="D")
    post_dates = pd.to_datetime(["2024-02-10", "2024-02-11", "2024-02-20"])
    history = pd.DataFrame({
        "date": list(pre_dates) + list(post_dates),
        "currency_pair": ["EUR_TND"] * (len(pre_dates) + len(post_dates)),
        "spot_mid": np.linspace(3.0, 3.4, len(pre_dates) + len(post_dates)),
    })
    accepted = pd.DataFrame({"family": ["importer"], "rho": [0.1], "sigma_Q": [0.2]})
    handoff = pd.DataFrame({
        "profile_id": [1, 2, 3],
        "currency_pair": ["EUR_TND"] * 3,
        "tenor_months": [6, 6, 6],
        "direction": ["outflow"] * 3,
        "E_t": [-0.10, -0.15, -0.12],
    })
    res = compute_rolling_market_performance(
        forward_backtest_long=forward,
        spot_history_long=history,
        accepted_profiles=accepted,
        stage_1_5_handoff=handoff,
        hedge_scenarios={"baseline_protection": 0.5},
        gamma_R_global={"baseline_protection": 0.5},
        stress_scenarios={"cip_base": 0},
        vol_window_days=30,
        run_id="x",
    )
    pre = history[history["date"] < pd.Timestamp("2024-02-10")].tail(31)["spot_mid"].to_numpy()
    sigma_annual = np.std(np.diff(np.log(pre)), ddof=1) * np.sqrt(252.0)
    expected = sigma_annual * np.sqrt(180.0 / 360.0)
    assert np.isclose(float(res["sigma_E"].iloc[0]), float(expected), atol=1e-10)
    assert res["market_row_status"].iloc[0] != "insufficient_vol_history"
