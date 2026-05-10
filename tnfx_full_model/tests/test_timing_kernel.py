from tnfx_full_model.timing_kernel import tenor_weights_for_profile


def test_gamma_weights_sum_to_one(config):
    df = tenor_weights_for_profile({"profile_id": 1, "c": 3.0}, config.model)
    assert len(df.groupby("timing_cv_scenario")) == 3
    assert (df.groupby("timing_cv_scenario").size() == 6).all()
    assert (df.groupby("timing_cv_scenario")["omega_t"].sum().sub(1).abs() < 1e-10).all()

