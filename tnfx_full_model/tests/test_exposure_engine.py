import pytest

from tnfx_full_model.exposure_engine import balanced_h_c_interval, compute_delta_cf_total, exposure_diagnostics, family_sign_check


def test_exposure_formulas(config):
    profile = {"family": "importer", "h_R": 0.1, "h_C": 0.5, "r": 0.1}
    delta = compute_delta_cf_total(profile)
    assert delta == 0.1 - 0.5 * 0.9
    assert family_sign_check("importer", delta, config.model["sign_thresholds"])
    diag = exposure_diagnostics(profile, config.model["sign_thresholds"])
    assert diag["delta_profit_total"] == delta / 0.1
    assert diag["profit_leverage_flag"] == (abs(delta / 0.1) > 10)


def test_conditional_h_c_interval_processor_trader(config):
    for family, tau_name, bounds in [("processor", "tau_P", [0.25, 0.80]), ("trader", "tau_T", [0.20, 0.95])]:
        h_R, r = 0.55, 0.08
        low, high = balanced_h_c_interval(h_R, r, config.model["sign_thresholds"][tau_name], bounds)
        h_C = (low + high) / 2
        delta = h_R - h_C * (1 - r)
        assert abs(delta) <= config.model["sign_thresholds"][tau_name]
        assert low <= high


def test_conditional_h_c_no_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        balanced_h_c_interval(0.5, 1.0, 0.2, [0.0, 1.0])
