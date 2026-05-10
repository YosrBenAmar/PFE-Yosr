from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_rolling_volatility(spot_history: pd.DataFrame, market: pd.DataFrame, window_days: int = 252, vol_method: str = "rolling") -> pd.DataFrame:
    if vol_method == "garch":
        raise NotImplementedError("vol_method='garch' is a placeholder and requires no extra dependencies here")
    hist = spot_history.sort_values(["currency_pair", "date"]).copy()
    hist["spot_mid"] = pd.to_numeric(hist["spot_mid"], errors="coerce")
    hist = hist[hist["spot_mid"].notna() & (hist["spot_mid"] > 0)].copy()
    min_rows = int(window_days) + 1
    counts = hist.groupby("currency_pair").size()
    for pair in market["currency_pair"].drop_duplicates():
        n = int(counts.get(pair, 0))
        if n < min_rows:
            raise ValueError(f"Insufficient spot history for {window_days}-day rolling volatility: {pair} has {n} rows.")
    hist["log_return"] = hist.groupby("currency_pair")["spot_mid"].transform(lambda s: np.log(s.astype(float)).diff())
    hist["rolling_std"] = hist.groupby("currency_pair")["log_return"].transform(
        lambda s: s.rolling(window_days, min_periods=2).std()
    )
    last = hist.groupby("currency_pair").tail(1).set_index("currency_pair")
    rows = []
    for row in market[["currency_pair", "tenor_months", "tenor_days"]].drop_duplicates().to_dict("records"):
        pair = row["currency_pair"]
        std = last.loc[pair, "rolling_std"] if pair in last.index else np.nan
        if pd.isna(std) or std <= 0:
            std = hist[hist["currency_pair"] == pair]["log_return"].std()
        if pd.isna(std) or std <= 0:
            std = 0.006
        sigma_annual = float(std) * np.sqrt(252.0)
        rows.append({**row, "sigma_E": sigma_annual * np.sqrt(float(row["tenor_days"]) / 360.0)})
    return pd.DataFrame(rows)
