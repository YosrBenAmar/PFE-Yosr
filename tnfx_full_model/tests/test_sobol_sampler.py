import numpy as np
import pytest

from tnfx_full_model.sobol_sampler import ConditionalIntervalError, _candidate_from_u, pilot_acceptance, sample_profiles, sobol_columns, sobol_points


def test_sobol_dimension_and_burn_in(config, small_run):
    assert len(sobol_columns()) == 15
    pts = sobol_points(2, small_run["sobol_dimension"], small_run["seed"], small_run["sobol_burn_in"])
    assert pts.shape == (2, 15)
    tables = sample_profiles(config.model, small_run)
    assert tables["Sobol_Meta"]["first_points_skipped"].iloc[0] == 1024


def test_family_selector_and_bounds(config, small_run):
    pilot = pilot_acceptance(config.model, small_run, "nominal")
    assert set(pilot["status"]).issubset({"pass", "warning", "validation_fail", "hard_fail"})
    tables = sample_profiles(config.model, small_run)
    accepted = tables["Accepted_Profiles"]
    assert not accepted.empty
    assert (accepted["sampling_weight"] > 0).all()
    for _, row in accepted.iterrows():
        spec = config.model["families"][row["family"]]
        for col in ["h_R", "h_C", "r", "beta", "fx_debt_service_share", "c", "lambda", "sigma_Q", "rho", "f"]:
            assert spec[col][0] <= row[col] <= spec[col][1]


def test_stratified_family_shares_close_to_prior(config, small_run):
    run = dict(small_run)
    run["target_profiles"] = 80
    run["sampling_mode"] = "stratified_by_family"
    tables = sample_profiles(config.model, run)
    shares = tables["Accepted_Profiles"]["family"].value_counts(normalize=True).to_dict()
    for family, target in config.model["family_priors"][run["primary_prior_mode"]].items():
        assert abs(shares.get(family, 0.0) - target) <= 0.03


def test_pooled_mode_still_works(config, small_run):
    run = dict(small_run)
    run["sampling_mode"] = "pooled_sobol"
    tables = sample_profiles(config.model, run)
    assert len(tables["Accepted_Profiles"]) == run["target_profiles"]


def test_conditional_interval_empty_rejection_reason(config):
    model = dict(config.model)
    model["families"] = dict(config.model["families"])
    model["families"]["trader"] = dict(config.model["families"]["trader"])
    model["families"]["trader"]["h_C"] = [0.0, 0.01]
    u = np.array([0.9, 0.5, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5])
    with pytest.raises(ConditionalIntervalError, match="conditional_h_C_interval_empty"):
        _candidate_from_u(u, 1024, "nominal", model, "trader")
