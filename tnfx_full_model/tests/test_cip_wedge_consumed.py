import numpy as np
import pandas as pd

from tnfx_full_model.rolling_market_performance import compute_rolling_market_performance


def test_cip_wedge_changes_carry_and_h_c():
    forward = pd.DataFrame([{ 
        "run_id": "x", "source_sheet": "EUR ASK Forward", "currency_pair": "EUR_TND", "currency": "EUR",
        "side": "ASK", "direction": "outflow", "tenor_months": 6, "transaction_type": "Export",
        "transaction_date": "2024-06-30", "hedge_transaction_date": "2024-01-15", "hedge_n_days": 180,
        "hedge_spot_rate": 3.30, "domestic_yield": 0.08, "foreign_yield": 0.03, "forward_rate": 3.34,
        "realized_spot": 3.25, "realized_forward_advantage": -0.09, "F_recomputed": 3.34,
        "cip_recalculation_error": 0.0, "cip_recalculation_status": "ok",
    }])
    history = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=300, freq="D"),
        "currency_pair": ["EUR_TND"] * 300,
        "spot_mid": 3.0 + 0.30 * np.sin(np.linspace(0, 20, 300)) + 0.05 * np.cos(np.linspace(0, 15, 300)),
    })
    accepted = pd.DataFrame({"family": ["importer"], "rho": [0.1], "sigma_Q": [0.2]})
    res = compute_rolling_market_performance(
        forward_backtest_long=forward,
        spot_history_long=history,
        accepted_profiles=accepted,
        hedge_scenarios={"baseline_protection": 0.5},
        gamma_R_global={"baseline_protection": 2.0},
        stress_scenarios={"cip_base": 0, "cip_plus_50bps": 50},
        vol_window_days=30,
        run_id="x",
    )
    base = res[res["forward_stress_scenario"] == "cip_base"].iloc[0]
    plus = res[res["forward_stress_scenario"] == "cip_plus_50bps"].iloc[0]
    assert base["carry_cost_used"] != plus["carry_cost_used"]
    assert base["h_c"] != plus["h_c"]