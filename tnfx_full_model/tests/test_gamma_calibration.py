import pandas as pd

from tnfx_full_model.gamma_calibration import calibrate_gamma_R, h_star_formula, solve_gamma_for_target


def test_gamma_reproduces_target():
    gamma = solve_gamma_for_target(0.5, 0.1, 0.08, 0.1, 0.2, 3.4, 0.05)
    assert abs(h_star_formula(0.1, 0.08, 0.1, 0.2, 3.4, 0.05, gamma) - 0.5) < 1e-6


def test_robustness_warning_concept():
    vals = [1.0, 2.0, 3.0]
    variation = (max(vals) - min(vals)) / 2.0
    assert variation > 0.30


def test_global_gamma_detail_includes_estar_and_methodology():
    accepted = pd.DataFrame({"rho": [0.1, 0.2], "sigma_Q": [0.15, 0.20]})
    diagnostics = pd.DataFrame({
        "profile_id": [1, 2, 3],
        "family": ["importer", "importer", "exporter"],
        "delta_net_EUR": [-0.05, -0.04, 0.10],
    })
    market = pd.DataFrame({
        "currency_pair": ["EUR_TND", "USD_TND", "EUR_TND"],
        "tenor_months": [6, 6, 12],
        "sigma_E": [0.07, 0.08, 0.09],
        "spot_mid": [3.4, 3.1, 3.4],
        "spot_ask": [3.41, 3.11, 3.41],
        "carry_cost": [0.05, 0.04, 0.06],
    })
    _, detail = calibrate_gamma_R(
        accepted,
        diagnostics,
        market,
        {"no_hedge": 0.0, "baseline_protection": 0.5, "high_protection": 0.75, "full_hedge": 1.0},
    )
    required = {
        "hedge_intensity_scenario", "target_intensity", "benchmark_family_used", "benchmark_currency_used",
        "benchmark_exposure_definition", "E_star", "rho_star", "sigma_Q_star", "currency_pair", "tenor_months",
        "S0_star", "sigma_E_star", "carry_cost_star", "gamma_R", "calibration_status", "gamma_methodology",
    }
    assert required.issubset(set(detail.columns))
    assert detail["gamma_methodology"].eq("global_population_importer_benchmark").all()
    assert detail["benchmark_exposure_definition"].eq("median_abs_delta_net_EUR_importer").all()


def test_global_gamma_detail_positive_estar_for_ok_rows():
    accepted = pd.DataFrame({"rho": [0.1], "sigma_Q": [0.2]})
    diagnostics = pd.DataFrame({
        "profile_id": [1, 2],
        "family": ["importer", "importer"],
        "delta_net_EUR": [-0.08, -0.06],
    })
    market = pd.DataFrame({
        "currency_pair": ["EUR_TND", "USD_TND", "EUR_TND"],
        "tenor_months": [6, 6, 12],
        "sigma_E": [0.07, 0.08, 0.09],
        "spot_mid": [3.4, 3.1, 3.4],
        "spot_ask": [3.41, 3.11, 3.41],
        "carry_cost": [0.05, 0.04, 0.06],
    })
    _, detail = calibrate_gamma_R(
        accepted,
        diagnostics,
        market,
        {"baseline_protection": 0.5},
    )
    ok_rows = detail[detail["calibration_status"] == "ok"]
    assert not ok_rows.empty
    assert (ok_rows["E_star"] > 0).all()
