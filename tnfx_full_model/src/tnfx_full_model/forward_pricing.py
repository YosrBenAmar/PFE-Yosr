from __future__ import annotations

import numpy as np
import pandas as pd


def cip_forward(S0: float, i_tnd: float, i_fcy: float, tenor_days: int) -> float:
    return float(S0) * (1.0 + float(i_tnd) * int(tenor_days) / 360.0) / (1.0 + float(i_fcy) * int(tenor_days) / 360.0)


def price_market_snapshot(market: pd.DataFrame, vol: pd.DataFrame | None = None, vol_method: str = "rolling", wedge_bps: float = 50.0) -> pd.DataFrame:
    rows = []
    for row in market.to_dict("records"):
        f_bid = cip_forward(row["spot_bid"], row["tnd_rate_bid"], row["fcy_rate_ask"], row["tenor_days"])
        f_ask = cip_forward(row["spot_ask"], row["tnd_rate_ask"], row["fcy_rate_bid"], row["tenor_days"])
        executable = row["forward_ask"] if not pd.isna(row["forward_ask"]) else f_ask
        s0 = row["spot_ask"]
        out = {**row, "F_CIP_bid": f_bid, "F_CIP_ask": f_ask, "F_executable": executable,
               "carry_cost": executable - s0, "forward_premium": executable / s0 - 1.0,
               "carry_cost_wedge": executable - s0 + (wedge_bps / 10000.0) * s0 * row["tenor_days"] / 360.0,
               "vol_method": vol_method}
        rows.append(out)
    snap = pd.DataFrame(rows)
    if vol is not None:
        snap = snap.merge(vol, on=["currency_pair", "tenor_months", "tenor_days"], how="left")
    if "sigma_E" not in snap:
        snap["sigma_E"] = 0.10
    snap["sigma_E"] = snap["sigma_E"].fillna(0.10)
    return snap


def executable_terms(row: dict, forward_bias: float = 0.0) -> dict:
    if row["direction"] == "outflow":
        s0 = row["spot_ask"]
        f_exec = row["forward_ask"] if not pd.isna(row.get("forward_ask")) else row["F_CIP_ask"]
        side = "ask"
    elif row["direction"] == "inflow":
        s0 = row["spot_bid"]
        f_exec = row["forward_bid"] if not pd.isna(row.get("forward_bid")) else row["F_CIP_bid"]
        side = "bid"
    else:
        raise ValueError(f"Unsupported direction {row['direction']}")
    return {
        "S0": s0, "F_executable": f_exec, "carry_cost": f_exec - s0,
        "forward_premium": f_exec / s0 - 1.0, "forward_bias": forward_bias,
        "pricing_side": side,
    }

