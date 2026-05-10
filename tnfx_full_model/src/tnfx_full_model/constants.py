SOBOL_COLUMNS = [
    "h_R",
    "h_C",
    "r",
    "beta",
    "fx_debt_service_share",
    "alpha_R_EUR",
    "alpha_C_EUR",
    "alpha_D_EUR",
    "c",
    "lambda",
    "sigma_Q",
    "rho",
    "f",
    "family_selector",
    "regime_state_selector",
]

FAMILIES = ["importer", "exporter", "processor", "trader"]
CURRENCIES = ["EUR", "USD"]
CURRENCY_PAIR = {"EUR": "EUR_TND", "USD": "USD_TND"}
REGIME_FLAGS = ["g_NR", "g_PAE", "g_TE", "g_AEO", "g_ACC", "g_EXT", "g_CIRC"]
STAGE1_MODULES = {
    "family_specs",
    "regime_engine",
    "sobol_sampler",
    "exposure_engine",
    "currency_layer",
    "timing_kernel",
    "stage15_handoff",
}
FORBIDDEN_COLUMNS = {"pi_F", "kappa", "delta_quantity_adjusted"}

