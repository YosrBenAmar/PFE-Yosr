from __future__ import annotations

from typing import Mapping


def decompose_currency(profile: Mapping, tolerance: float = 1e-10) -> dict:
    h_R = float(profile["h_R"])
    h_C = float(profile["h_C"])
    r = float(profile["r"])
    beta = float(profile["beta"])
    alpha_R_EUR = float(profile["alpha_R_EUR"])
    alpha_C_EUR = float(profile["alpha_C_EUR"])
    alpha_D_EUR = float(profile["alpha_D_EUR"])
    alpha_R_USD = 1.0 - alpha_R_EUR
    alpha_C_USD = 1.0 - alpha_C_EUR
    alpha_D_USD = 1.0 - alpha_D_EUR
    h_R_EUR, h_R_USD = h_R * alpha_R_EUR, h_R * alpha_R_USD
    h_C_EUR, h_C_USD = h_C * alpha_C_EUR, h_C * alpha_C_USD
    delta_CF_EUR = h_R_EUR - h_C_EUR * (1.0 - r)
    delta_CF_USD = h_R_USD - h_C_USD * (1.0 - r)
    delta_CF_total = h_R - h_C * (1.0 - r)
    if abs(delta_CF_EUR + delta_CF_USD - delta_CF_total) >= tolerance:
        raise ValueError("Currency delta_CF identity failed")
    if int(profile["g_EXT"]) == 1:
        debt_eur = float(profile["fx_debt_service_share"]) * alpha_D_EUR
        debt_usd = float(profile["fx_debt_service_share"]) * alpha_D_USD
    else:
        debt_eur = 0.0
        debt_usd = 0.0
    delta_op_eff_EUR = (1.0 - beta) * delta_CF_EUR
    delta_op_eff_USD = (1.0 - beta) * delta_CF_USD
    delta_net_EUR = delta_op_eff_EUR - debt_eur
    delta_net_USD = delta_op_eff_USD - debt_usd
    delta_net_total = delta_net_EUR + delta_net_USD
    gross = abs(delta_net_EUR) + abs(delta_net_USD)
    net = abs(delta_net_total)
    return {
        "alpha_R_USD": alpha_R_USD, "alpha_C_USD": alpha_C_USD, "alpha_D_USD": alpha_D_USD,
        "h_R_EUR": h_R_EUR, "h_R_USD": h_R_USD, "h_C_EUR": h_C_EUR, "h_C_USD": h_C_USD,
        "delta_CF_EUR": delta_CF_EUR, "delta_CF_USD": delta_CF_USD,
        "delta_op_eff_EUR": delta_op_eff_EUR, "delta_op_eff_USD": delta_op_eff_USD,
        "fx_debt_service_share_EUR": debt_eur, "fx_debt_service_share_USD": debt_usd,
        "delta_net_EUR": delta_net_EUR, "delta_net_USD": delta_net_USD,
        "delta_net_total": delta_net_total,
        "currency_mismatch_gap": gross - net,
        "currency_mismatch_flag": (gross - net) > 0.10,
        "currency_active_EUR": abs(delta_net_EUR) >= 0.005,
        "currency_active_USD": abs(delta_net_USD) >= 0.005,
    }

