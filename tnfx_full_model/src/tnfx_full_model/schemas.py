ACCEPTED_PROFILE_COLUMNS = [
    "profile_id", "sobol_index", "prior_mode", "family", "subtype",
    "g_NR", "g_PAE", "g_TE", "g_AEO", "g_ACC", "g_EXT", "g_CIRC",
    "h_R", "h_C", "r", "beta", "fx_debt_service_share",
    "alpha_R_EUR", "alpha_R_USD", "alpha_C_EUR", "alpha_C_USD",
    "alpha_D_EUR", "alpha_D_USD", "c", "lambda", "sigma_Q", "rho", "f",
    "target_prior_family_share", "accepted_sample_family_share", "sampling_weight",
]
# f: frequency proxy for exposure crystallisation events.
# Carried in Sobol dim 13 to preserve 15-D low-discrepancy structure.
# Not consumed by Stage 1, 1.5, or Stage 2 computations.
# Reserved for Stage 3 intra-period settlement frequency modeling.

HANDOFF_COLUMNS = [
    "profile_id", "family", "subtype", "currency", "currency_pair",
    "timing_cv_scenario", "timing_CV", "tenor_months", "h_R_k", "h_C_k",
    "alpha_R_k", "alpha_C_k", "alpha_D_k", "fx_debt_service_share_k",
    "delta_CF_k", "delta_op_eff_k", "delta_net_k", "omega_t", "E_t",
    "Q_plus", "Q_minus", "lambda", "H_t", "direction", "sigma_Q", "rho", "c",
    "g_NR", "g_PAE", "g_TE", "g_AEO", "g_ACC", "g_EXT", "g_CIRC",
]

MARKET_COLUMNS = [
    "valuation_date", "currency_pair", "tenor_months", "tenor_days",
    "spot_bid", "spot_ask", "spot_mid", "tnd_rate_bid", "tnd_rate_ask",
    "fcy_rate_bid", "fcy_rate_ask", "forward_bid", "forward_ask",
    "realized_future_date", "realized_future_spot_bid", "realized_future_spot_ask",
]
