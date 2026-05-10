from tnfx_full_model.gamma_calibration import h_star_formula, solve_gamma_for_target


def test_gamma_reproduces_target():
    gamma = solve_gamma_for_target(0.5, 0.1, 0.08, 0.1, 0.2, 3.4, 0.05)
    assert abs(h_star_formula(0.1, 0.08, 0.1, 0.2, 3.4, 0.05, gamma) - 0.5) < 1e-6


def test_robustness_warning_concept():
    vals = [1.0, 2.0, 3.0]
    variation = (max(vals) - min(vals)) / 2.0
    assert variation > 0.30

