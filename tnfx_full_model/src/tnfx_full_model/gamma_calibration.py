from __future__ import annotations

import numpy as np
import pandas as pd


def h_star_formula(E: float, sigma_E: float, rho: float, sigma_Q: float, S0: float, carry_cost: float, gamma_R: float, forward_bias: float = 0.0) -> float:
    sign_e = 1.0 if E >= 0 else -1.0
    denom = 2.0 * sigma_E ** 2 * abs(E) * S0
    cost_term = 0.0 if denom == 0 else gamma_R * (carry_cost + forward_bias) / denom
    return 1.0 + rho * sigma_Q / (sigma_E * sign_e) - cost_term


def solve_gamma_for_target(target: float, E: float, sigma_E: float, rho: float, sigma_Q: float, S0: float, carry_cost: float, forward_bias: float = 0.0) -> float:
    denom = carry_cost + forward_bias
    numerator = (1.0 + rho * sigma_Q / sigma_E - target) * (2.0 * sigma_E ** 2 * abs(E) * S0)
    if abs(denom) < 1e-12:
        raise ValueError("Cannot calibrate gamma_R with zero carry denominator")
    gamma = numerator / denom
    if gamma <= 0:
        raise ValueError("Calibrated gamma_R is non-positive")
    return float(gamma)


def calibrate_gamma_R(accepted: pd.DataFrame, diagnostics: pd.DataFrame, market_snapshot: pd.DataFrame, targets: dict[str, float]) -> tuple[dict[str, float], pd.DataFrame]:
    diag = diagnostics.set_index("profile_id")
    importer_mask = diag["family"] == "importer" if "family" in diag.columns else pd.Series(dtype=bool)
    if "delta_net_EUR" in diag.columns and importer_mask.any():
        E = float(diag.loc[importer_mask, "delta_net_EUR"].abs().median())
        benchmark_family_used = "importer"
        benchmark_exposure_definition = "median_abs_delta_net_EUR_importer"
    elif "delta_net_EUR" in diag.columns:
        E = float(diag["delta_net_EUR"].abs().median())
        benchmark_family_used = "all_families"
        benchmark_exposure_definition = "median_abs_delta_net_EUR_all_families"
    else:
        E = np.nan
        benchmark_family_used = "all_families"
        benchmark_exposure_definition = "delta_net_EUR_missing"
    rho = float(accepted["rho"].median())
    sigma_Q = float(accepted["sigma_Q"].median())
    bench_keys = [("EUR_TND", 6), ("USD_TND", 6), ("EUR_TND", 12)]
    details = []
    calibrated = {}
    for scenario, tau in targets.items():
        if tau in {0.0, 1.0}:
            continue
        gammas = []
        for pair, tenor in bench_keys:
            row = market_snapshot[(market_snapshot["currency_pair"] == pair) & (market_snapshot["tenor_months"] == tenor)]
            if row.empty:
                continue
            r = row.iloc[0]
            carry_cost = float(r["carry_cost"])
            forward_bias_placeholder = 0.0
            s0_star = float(r.get("spot_ask", r["spot_mid"]))
            sigma_e_star = float(r["sigma_E"])
            gamma = np.nan
            status = "ok"
            if not np.isfinite(E) or E <= 0:
                status = "invalid_benchmark_exposure"
            elif carry_cost + forward_bias_placeholder <= 0:
                status = "negative_carry_cost_skipped"
            else:
                try:
                    # Use spot_ask for calibration benchmark: importer outflows execute at ASK.
                    # Falls back to spot_mid for backward compatibility if ASK not present.
                    gamma = solve_gamma_for_target(
                        tau, E, sigma_e_star, rho, sigma_Q, s0_star, carry_cost
                    )
                    gammas.append(gamma)
                except ValueError as exc:
                    status = str(exc)
            details.append({
                "hedge_intensity_scenario": scenario,
                "target_intensity": tau,
                "benchmark_family_used": benchmark_family_used,
                "benchmark_currency_used": "EUR",
                "benchmark_exposure_definition": benchmark_exposure_definition,
                "E_star": E,
                "rho_star": rho,
                "sigma_Q_star": sigma_Q,
                "currency_pair": pair,
                "tenor_months": tenor,
                "S0_star": s0_star,
                "sigma_E_star": sigma_e_star,
                "carry_cost_star": carry_cost,
                "gamma_R": gamma,
                "calibration_status": status,
                "status": status,
                "gamma_methodology": "global_population_importer_benchmark",
            })
        valid = [g for g in gammas if np.isfinite(g) and g > 0]
        if not valid:
            calibrated[scenario] = 1.0
        else:
            calibrated[scenario] = float(np.median(valid))
    detail_df = pd.DataFrame(details)
    if not detail_df.empty:
        detail_df["variation_pct"] = detail_df.groupby("hedge_intensity_scenario")["gamma_R"].transform(
            lambda s: (s.max() - s.min()) / s.median() if s.median() and not pd.isna(s.median()) else np.nan
        )
        detail_df["robustness_warning"] = detail_df["variation_pct"] > 0.30
    return calibrated, detail_df
