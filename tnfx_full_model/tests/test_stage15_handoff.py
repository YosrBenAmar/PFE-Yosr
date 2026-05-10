from tnfx_full_model.sobol_sampler import sample_profiles
from tnfx_full_model.stage15_handoff import build_handoff
from tnfx_full_model.timing_kernel import build_tenor_weights


def test_handoff_rows(config, small_run):
    tables = sample_profiles(config.model, small_run)
    weights = build_tenor_weights(tables["Accepted_Profiles"], config.model)
    handoff = build_handoff(tables["Accepted_Profiles"], tables["BM_Exposure_Diagnostics"], weights)
    counts = handoff.groupby("profile_id").size()
    assert counts.isin([18, 36]).all()
    assert not handoff[["currency_pair", "tenor_months", "direction"]].isna().any().any()
    assert set(tables["Accepted_Profiles"]["profile_id"]) == set(handoff["profile_id"])


def test_inactive_profiles_are_separate(config, small_run):
    tables = sample_profiles(config.model, small_run)
    inactive = tables["Inactive_Profiles"]
    if not inactive.empty:
        assert (inactive["inactive_reason"] == "no_active_currency").all()
