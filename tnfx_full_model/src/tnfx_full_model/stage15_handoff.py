from __future__ import annotations

import pandas as pd

from .constants import CURRENCY_PAIR, REGIME_FLAGS
from .schemas import HANDOFF_COLUMNS


def build_handoff(accepted: pd.DataFrame, diagnostics: pd.DataFrame, tenor_weights: pd.DataFrame, active_currency_threshold: float = 0.005) -> pd.DataFrame:
    diag = diagnostics.set_index("profile_id")
    rows = []
    for profile in accepted.to_dict("records"):
        profile_diag = diag.loc[profile["profile_id"]]
        for currency in ["EUR", "USD"]:
            delta_net = float(profile_diag[f"delta_net_{currency}"])
            if abs(delta_net) < float(active_currency_threshold):
                continue
            for tw in tenor_weights[tenor_weights["profile_id"] == profile["profile_id"]].to_dict("records"):
                e_t = delta_net * float(tw["omega_t"])
                if e_t == 0:
                    continue
                direction = "inflow" if e_t > 0 else "outflow"
                row = {
                    "profile_id": profile["profile_id"], "family": profile["family"], "subtype": profile["subtype"],
                    "currency": currency, "currency_pair": CURRENCY_PAIR[currency],
                    "timing_cv_scenario": tw["timing_cv_scenario"], "timing_CV": tw["timing_CV"],
                    "tenor_months": tw["tenor_months"], "h_R_k": profile_diag[f"h_R_{currency}"],
                    "h_C_k": profile_diag[f"h_C_{currency}"], "alpha_R_k": profile[f"alpha_R_{currency}"],
                    "alpha_C_k": profile[f"alpha_C_{currency}"], "alpha_D_k": profile[f"alpha_D_{currency}"],
                    "fx_debt_service_share_k": profile_diag[f"fx_debt_service_share_{currency}"],
                    "delta_CF_k": profile_diag[f"delta_CF_{currency}"],
                    "delta_op_eff_k": profile_diag[f"delta_op_eff_{currency}"],
                    "delta_net_k": delta_net, "omega_t": tw["omega_t"], "E_t": e_t,
                    "Q_plus": max(e_t, 0.0), "Q_minus": max(-e_t, 0.0),
                    "lambda": profile["lambda"], "H_t": profile["lambda"] * abs(e_t),
                    "direction": direction, "sigma_Q": profile["sigma_Q"], "rho": profile["rho"], "c": profile["c"],
                }
                for flag in REGIME_FLAGS:
                    row[flag] = profile[flag]
                rows.append(row)
    return pd.DataFrame(rows, columns=HANDOFF_COLUMNS)
