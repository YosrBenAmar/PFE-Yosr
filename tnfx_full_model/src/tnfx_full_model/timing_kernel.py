from __future__ import annotations

import pandas as pd
from scipy.stats import gamma


def timing_scenarios(model_config: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"timing_cv_scenario": k, "timing_CV": v, "role": "exposure_timing_dispersion"}
         for k, v in model_config["timing_cv_scenarios"].items()]
    )


def tenor_weights_for_profile(profile: dict, model_config: dict) -> pd.DataFrame:
    rows = []
    tenors = model_config["tenors_months"]
    for scenario, cv in model_config["timing_cv_scenarios"].items():
        k_gamma = 1.0 / (float(cv) ** 2)
        theta_gamma = float(profile["c"]) * (float(cv) ** 2)
        raw = [float(gamma.pdf(t, a=k_gamma, scale=theta_gamma)) for t in tenors]
        if sum(raw) <= 0:
            raw = [1.0 for _ in tenors]
        weights = [x / sum(raw) for x in raw]
        for t, rw, omega in zip(tenors, raw, weights):
            rows.append({
                "profile_id": profile["profile_id"], "timing_cv_scenario": scenario,
                "timing_CV": float(cv), "tenor_months": int(t),
                "k_gamma": k_gamma, "theta_gamma": theta_gamma,
                "raw_weight": rw, "omega_t": omega, "weight_sum_check": sum(weights),
            })
    return pd.DataFrame(rows)


def build_tenor_weights(profiles: pd.DataFrame, model_config: dict) -> pd.DataFrame:
    return pd.concat([tenor_weights_for_profile(row, model_config) for row in profiles.to_dict("records")], ignore_index=True)

