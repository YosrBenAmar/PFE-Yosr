import numpy as np
import pandas as pd

from tnfx_full_model.rolling_market_performance import compute_oos_market_performance


def _synthetic_forward():
    rows = []
    for year in range(2008, 2026):
        rows.append({
            "run_id": "x", "source_sheet": "EUR ASK Forward", "currency_pair": "EUR_TND", "currency": "EUR",
            "side": "ASK", "direction": "outflow", "tenor_months": 6, "transaction_type": "Export",
            "transaction_date": f"{year}-06-30", "hedge_transaction_date": f"{year}-01-15", "hedge_n_days": 180,
            "hedge_spot_rate": 3.0 + 0.01 * (year - 2008), "domestic_yield": 0.08, "foreign_yield": 0.03,
            "forward_rate": 3.1 + 0.02 * (year - 2008), "realized_spot": 3.05 + 0.01 * (year - 2008),
            "realized_forward_advantage": -0.05 + 0.002 * (year - 2008), "F_recomputed": 3.1,
            "cip_recalculation_error": 0.0, "cip_recalculation_status": "ok",
        })
    return pd.DataFrame(rows)


def _synthetic_history():
    dates = pd.date_range("2007-01-01", "2025-12-31", freq="D")
    return pd.DataFrame({
        "date": dates,
        "currency_pair": ["EUR_TND"] * len(dates),
        "spot_mid": 3.0 + np.sin(np.linspace(0, 20, len(dates))) * 0.05 + np.linspace(0, 0.4, len(dates)),
    })


def _accepted():
    return pd.DataFrame({"family": ["importer", "exporter"], "rho": [0.1, 0.15], "sigma_Q": [0.2, 0.25]})


def test_oos_uses_split_specific_gamma():
    oos, _ = compute_oos_market_performance(
        forward_backtest_long=_synthetic_forward(),
        spot_history_long=_synthetic_history(),
        accepted_profiles=_accepted(),
        hedge_scenarios={"no_hedge": 0.0, "low_protection": 0.25, "baseline_protection": 0.5, "high_protection": 0.75, "full_hedge": 1.0},
        stress_scenarios={"cip_base": 0, "cip_plus_50bps": 50},
        vol_window_days=30,
        run_id="x",
    )
    rows = oos[(oos["hedge_intensity_scenario"] == "baseline_protection") & (oos["forward_stress_scenario"] == "cip_base")]
    g2015 = rows[rows["test_start"] == pd.Timestamp("2015-01-01")]["gamma_R_used"].dropna().iloc[0]
    g2017 = rows[rows["test_start"] == pd.Timestamp("2017-01-01")]["gamma_R_used"].dropna().iloc[0]
    assert not np.isclose(g2015, g2017)


def test_first_split_is_2008_2013_to_2014():
    _, cal = compute_oos_market_performance(
        forward_backtest_long=_synthetic_forward(),
        spot_history_long=_synthetic_history(),
        accepted_profiles=_accepted(),
        hedge_scenarios={"no_hedge": 0.0, "low_protection": 0.25, "baseline_protection": 0.5, "high_protection": 0.75, "full_hedge": 1.0},
        stress_scenarios={"cip_base": 0, "cip_plus_50bps": 50},
        vol_window_days=30,
        run_id="x",
    )
    first = cal.sort_values(["split_id", "hedge_intensity_scenario"]).iloc[0]
    assert pd.to_datetime(first["train_start"]).strftime("%Y-%m-%d") == "2008-01-01"
    assert pd.to_datetime(first["train_end"]).strftime("%Y-%m-%d") == "2013-12-31"
    assert pd.to_datetime(first["test_start"]).strftime("%Y-%m-%d") == "2014-01-01"


def test_oos_gamma_table_has_methodology_disclosure():
    _, cal = compute_oos_market_performance(
        forward_backtest_long=_synthetic_forward(),
        spot_history_long=_synthetic_history(),
        accepted_profiles=_accepted(),
        hedge_scenarios={"no_hedge": 0.0, "low_protection": 0.25, "baseline_protection": 0.5, "high_protection": 0.75, "full_hedge": 1.0},
        stress_scenarios={"cip_base": 0, "cip_plus_50bps": 50},
        vol_window_days=30,
        run_id="x",
    )
    required = {
        "gamma_methodology",
        "benchmark_exposure_definition",
        "calibration_currency_pair",
        "calibration_tenor_months",
        "n_training_observations",
        "calibration_status",
    }
    assert required.issubset(set(cal.columns))


def test_oos_gamma_table_labels_market_side_fixed_unit_exposure():
    _, cal = compute_oos_market_performance(
        forward_backtest_long=_synthetic_forward(),
        spot_history_long=_synthetic_history(),
        accepted_profiles=_accepted(),
        hedge_scenarios={"no_hedge": 0.0, "low_protection": 0.25, "baseline_protection": 0.5, "high_protection": 0.75, "full_hedge": 1.0},
        stress_scenarios={"cip_base": 0, "cip_plus_50bps": 50},
        vol_window_days=30,
        run_id="x",
    )
    active = cal[cal["hedge_intensity_scenario"].isin(["low_protection", "baseline_protection", "high_protection"])]
    assert not active.empty
    assert active["gamma_methodology"].eq("split_specific_market_side").all()
    assert active["benchmark_exposure_definition"].eq("fixed_unit_exposure").all()
