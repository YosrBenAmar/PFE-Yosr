from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


TENOR_BUCKETS = np.array([1, 2, 3, 6, 9, 12], dtype=int)
DEFAULT_THRESHOLDS = {
    "pure_side_share": 0.85,
    "balanced_low": 0.30,
    "balanced_high": 0.70,
    "processor_volume_tnd": 5_000_000.0,
}
CLIENT_UNIVERSE_COLUMNS = [
    "client",
    "n_deals",
    "total_tnd",
    "buy_share",
    "sell_share",
    "eur_share",
    "usd_share",
    "observed_hedge_ratio",
    "mean_tenor_days",
    "n_hedge_deals",
    "family",
    "included_in_tests",
]

STRATEGY_RANKING_CANDIDATES = {
    "rank": ["rank", "ranking", "Rank"],
    "currency_pair": ["currency_pair", "CurrencyPair", "currency"],
    "tenor_months": ["tenor_months", "tenor", "Tenor"],
    "side": ["side", "Side"],
    "forward_stress_scenario": ["forward_stress_scenario", "cip_variant", "stress_scenario"],
    "hedge_intensity_scenario": ["hedge_intensity_scenario", "scenario", "best_scenario", "intensity_scenario"],
}

HEDGE_RECOMMENDATIONS_CANDIDATES = {
    "family": ["family", "Family"],
    "currency_pair": ["currency_pair", "CurrencyPair"],
    "tenor_months": ["tenor_months", "tenor"],
    "timing_cv_scenario": ["timing_cv_scenario", "cv_scenario"],
    "selected_hedge_intensity_scenario": [
        "selected_hedge_intensity_scenario",
        "hedge_intensity_scenario",
        "scenario",
        "best_scenario",
    ],
    "recommended_hedge_ratio": ["recommended_hedge_ratio", "h_recommended", "h_c"],
}

STAGE15_HANDOFF_CANDIDATES = {
    "family": ["family", "Family"],
    "c": ["c", "c_months", "exposure_cycle", "c_i"],
    "omega_t": ["omega_t", "weight", "tenor_weight", "w"],
    "tenor_months": ["tenor_months", "tenor"],
    "direction": ["direction", "Direction"],
}

ROLLING_SUMMARY_CANDIDATES = {
    "currency_pair": ["currency_pair"],
    "tenor_months": ["tenor_months", "tenor"],
    "side": ["side"],
    "hedge_intensity_scenario": ["hedge_intensity_scenario", "scenario"],
    "forward_stress_scenario": ["forward_stress_scenario", "cip_variant"],
    "mean_forward_advantage": ["mean_forward_advantage_bps", "mean_forward_advantage", "mean_carry", "mean_pnl"],
}


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(f"[validation] {msg}")


def _snap_tenor_months(months: float | int | None, default: int = 3) -> int:
    if months is None or (isinstance(months, float) and not np.isfinite(months)):
        return int(default)
    m = float(months)
    idx = int(np.argmin(np.abs(TENOR_BUCKETS.astype(float) - m)))
    return int(TENOR_BUCKETS[idx])


def _normalize_instrument(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    mapped = pd.Series(np.where(s.isin(["terme", "forward"]), "hedge", np.where(s == "spot", "spot", "other")), index=series.index)
    return mapped


def _currency_pair_from_colombus(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    return s.map({"EURTND": "EUR_TND", "USDTND": "USD_TND"})


def _side_from_type(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.map({"buy": "ASK", "sell": "BID"})


def _direction_from_type(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.map({"buy": "outflow", "sell": "inflow"})


def _require_columns(df: pd.DataFrame, required: list[str], source_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def _resolve_columns(df: pd.DataFrame, candidates: dict[str, list[str]], source_name: str) -> dict[str, str]:
    """Resolve logical column names to actual DataFrame columns.

    Matching is case-insensitive and whitespace-insensitive.
    """

    def _norm(name: str) -> str:
        return "".join(str(name).split()).lower()

    normalized_actual: dict[str, str] = {}
    for col in df.columns:
        key = _norm(str(col))
        if key not in normalized_actual:
            normalized_actual[key] = str(col)

    resolved: dict[str, str] = {}
    for logical_name, options in candidates.items():
        match = None
        for opt in options:
            key = _norm(opt)
            if key in normalized_actual:
                match = normalized_actual[key]
                break
        if match is None:
            raise ValueError(
                f"{source_name} has no column matching '{logical_name}'. "
                f"Looked for {options}. Actual columns: {list(df.columns)}"
            )
        resolved[logical_name] = match
    return resolved


def _make_realized_spot_lookup(spot_history: pd.DataFrame) -> Callable[[pd.Timestamp, str], float]:
    hist = spot_history.copy()
    hist = hist.sort_values("date").reset_index(drop=True)
    min_date = pd.to_datetime(hist["date"], errors="coerce").min()
    max_date = pd.to_datetime(hist["date"], errors="coerce").max()
    if pd.isna(min_date) or pd.isna(max_date):
        raise ValueError("IB spot history has invalid date range.")

    pair_to_col = {"EUR_TND": "EURTND", "USD_TND": "USDTND"}
    full_idx = pd.date_range(min_date, max_date, freq="D")
    prepared: dict[str, pd.Series] = {}
    for pair, col in pair_to_col.items():
        if col not in hist.columns:
            continue
        ser = pd.Series(pd.to_numeric(hist[col], errors="coerce").values, index=pd.to_datetime(hist["date"]))
        ser = ser[~ser.index.duplicated(keep="last")].sort_index()
        prepared[pair] = ser.reindex(full_idx).ffill().bfill()

    def lookup(date: pd.Timestamp, currency_pair: str) -> float:
        cp = str(currency_pair).upper().strip()
        if cp not in prepared:
            raise ValueError(f"Unsupported currency_pair for realized spot lookup: {currency_pair}")
        ts = pd.to_datetime(date, errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"Invalid date for realized spot lookup: {date}")
        ts = ts.normalize()
        if ts < min_date:
            raise ValueError(
                f"Requested date {ts.date()} is before first IB observation {min_date.date()}."
            )
        ts = min(ts, max_date)
        val = prepared[cp].loc[ts]
        if not np.isfinite(val):
            raise ValueError(f"Missing realized spot for {cp} at {ts.date()}.")
        return float(val)

    return lookup


def _load_colombus(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (transactions_df, spot_history_df). Applies all filters and renames."""
    if not path.exists():
        raise FileNotFoundError(
            f"Colombus workbook not found at {path}. Place the file at "
            "data/external/colombus_transactions.xlsx and retry."
        )

    tx = pd.read_excel(path, sheet_name="Transactions", header=1)
    ib = pd.read_excel(path, sheet_name="IB", header=1)

    tx = tx.loc[:, [c for c in tx.columns if not str(c).startswith("Unnamed:")]].copy()
    ib = ib.loc[:, [c for c in ib.columns if not str(c).startswith("Unnamed:")]].copy()

    tx_required = [
        "Client",
        "Transaction date",
        "Value date",
        "Currency",
        "Type",
        "Instrument",
        "Amount",
        "Rate",
        "Mid market",
        "Spot",
        "Amount in TND",
    ]
    _require_columns(tx, tx_required, "Transactions sheet")

    ib_rename = {c: str(c).strip().lower() for c in ib.columns}
    ib = ib.rename(columns=ib_rename)
    if "date" not in ib.columns:
        raise ValueError("IB sheet must include a Date column (may include surrounding spaces).")
    ib = ib.rename(columns={"date": "date"})
    ib_keep = ["date", "usdtnd", "eurtnd"]
    missing_ib = [c for c in ib_keep if c not in ib.columns]
    if missing_ib:
        raise ValueError(f"IB sheet is missing required columns: {missing_ib}")
    ib = ib[ib_keep].copy()
    ib = ib.rename(columns={"usdtnd": "USDTND", "eurtnd": "EURTND"})
    ib["date"] = pd.to_datetime(ib["date"], errors="coerce")
    ib["USDTND"] = pd.to_numeric(ib["USDTND"], errors="coerce")
    ib["EURTND"] = pd.to_numeric(ib["EURTND"], errors="coerce")
    ib = ib.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    tx["Client"] = tx["Client"].astype(str).str.strip()
    tx["Transaction date"] = pd.to_datetime(tx["Transaction date"], errors="coerce")
    tx["Value date"] = pd.to_datetime(tx["Value date"], errors="coerce")
    tx["Amount"] = pd.to_numeric(tx["Amount"], errors="coerce")
    tx["Rate"] = pd.to_numeric(tx["Rate"], errors="coerce")
    tx["Mid market"] = pd.to_numeric(tx["Mid market"], errors="coerce")
    tx["Spot"] = pd.to_numeric(tx["Spot"], errors="coerce")
    tx["Amount in TND"] = pd.to_numeric(tx["Amount in TND"], errors="coerce")
    tx["Currency"] = tx["Currency"].astype(str).str.upper().str.strip()
    tx["Type"] = tx["Type"].astype(str).str.strip()
    tx["Instrument"] = tx["Instrument"].astype(str).str.strip()

    tx = tx[tx["Currency"].isin(["EURTND", "USDTND"])].copy()
    instrument_lower = tx["Instrument"].astype(str).str.strip().str.lower()
    tx = tx[~instrument_lower.isin(["option"])].copy()
    instrument_lower = tx["Instrument"].astype(str).str.strip().str.lower()
    tx = tx[instrument_lower.isin(["spot", "terme", "forward"])].copy()
    tx["instrument_class"] = _normalize_instrument(tx["Instrument"])
    tx = tx[tx["instrument_class"].isin(["spot", "hedge"])].copy()
    tx["currency_pair"] = _currency_pair_from_colombus(tx["Currency"])
    tx["side"] = _side_from_type(tx["Type"])
    tx["direction"] = _direction_from_type(tx["Type"])
    tx["Amount_TND"] = tx["Amount in TND"]
    tx["Amount_TND"] = tx["Amount_TND"].where(tx["Amount_TND"].notna(), tx["Amount"] * tx["Rate"])
    tx["tenor_days"] = (tx["Value date"] - tx["Transaction date"]).dt.days
    tx = tx.dropna(subset=["Client", "Transaction date", "Currency", "Type", "Instrument", "Amount", "Rate", "Amount_TND", "currency_pair", "side"]).reset_index(drop=True)

    return tx, ib


def _classify_clients(tx: pd.DataFrame, min_deals: int, thresholds: dict) -> pd.DataFrame:
    """Test 1 input. Returns one row per client with per-client features and family labels."""
    working = tx.copy()
    working["is_buy"] = working["Type"].astype(str).str.lower().eq("buy")
    working["is_sell"] = working["Type"].astype(str).str.lower().eq("sell")
    working["is_hedge"] = working["instrument_class"].eq("hedge")
    working["is_eur"] = working["Currency"].eq("EURTND")
    working["is_usd"] = working["Currency"].eq("USDTND")

    grp = working.groupby("Client", sort=False)
    universe = grp["Amount_TND"].sum().rename("total_tnd").reset_index().rename(columns={"Client": "client"})
    universe["n_deals"] = grp.size().values
    universe["buy_tnd"] = working.loc[working["is_buy"]].groupby("Client")["Amount_TND"].sum().reindex(universe["client"]).fillna(0.0).values
    universe["sell_tnd"] = working.loc[working["is_sell"]].groupby("Client")["Amount_TND"].sum().reindex(universe["client"]).fillna(0.0).values
    universe["eur_tnd"] = working.loc[working["is_eur"]].groupby("Client")["Amount_TND"].sum().reindex(universe["client"]).fillna(0.0).values
    universe["usd_tnd"] = working.loc[working["is_usd"]].groupby("Client")["Amount_TND"].sum().reindex(universe["client"]).fillna(0.0).values
    universe["hedge_tnd"] = working.loc[working["is_hedge"]].groupby("Client")["Amount_TND"].sum().reindex(universe["client"]).fillna(0.0).values
    universe["n_hedge_deals"] = working.loc[working["is_hedge"]].groupby("Client").size().reindex(universe["client"]).fillna(0).astype(int).values
    universe["mean_tenor_days"] = (
        pd.to_numeric(working.loc[working["is_hedge"], "tenor_days"], errors="coerce")
        .groupby(working.loc[working["is_hedge"], "Client"])
        .mean()
        .reindex(universe["client"])
        .values
    )

    denom = universe["total_tnd"].replace(0, np.nan)
    universe["buy_share"] = (universe["buy_tnd"] / denom).fillna(0.0)
    universe["sell_share"] = (universe["sell_tnd"] / denom).fillna(0.0)
    universe["eur_share"] = (universe["eur_tnd"] / denom).fillna(0.0)
    universe["usd_share"] = (universe["usd_tnd"] / denom).fillna(0.0)
    universe["observed_hedge_ratio"] = (universe["hedge_tnd"] / denom).fillna(0.0)
    universe["included_in_tests"] = universe["n_deals"] >= int(min_deals)

    pure = float(thresholds["pure_side_share"])
    balanced_low = float(thresholds["balanced_low"])
    balanced_high = float(thresholds["balanced_high"])
    proc_tnd = float(thresholds["processor_volume_tnd"])

    def _label(row: pd.Series) -> str:
        if not bool(row["included_in_tests"]):
            return "excluded_low_activity"
        if row["buy_share"] >= pure:
            return "importer"
        if row["sell_share"] >= pure:
            return "exporter"
        if balanced_low <= row["buy_share"] <= balanced_high:
            return "processor" if row["total_tnd"] >= proc_tnd else "trader"
        return "mixed"

    universe["family"] = universe.apply(_label, axis=1)
    universe = universe[
        [
            "client",
            "n_deals",
            "total_tnd",
            "buy_share",
            "sell_share",
            "eur_share",
            "usd_share",
            "observed_hedge_ratio",
            "mean_tenor_days",
            "n_hedge_deals",
            "family",
            "included_in_tests",
        ]
    ].copy()
    return universe


def _test1_family_distribution(client_universe: pd.DataFrame, family_summary: pd.DataFrame) -> dict:
    """Compares observed family distribution against the three macro-anchor priors."""
    priors = {
        "nominal": {"importer": 0.35, "exporter": 0.20, "processor": 0.30, "trader": 0.15},
        "firm_count": {"importer": 0.45, "exporter": 0.15, "processor": 0.25, "trader": 0.15},
        "trade_flow": {"importer": 0.30, "exporter": 0.20, "processor": 0.40, "trader": 0.10},
    }
    families = ["importer", "exporter", "processor", "trader"]
    used = client_universe[client_universe["included_in_tests"] & client_universe["family"].isin(families)].copy()
    totals_clients = max(int(len(used)), 1)
    totals_volume = max(float(used["total_tnd"].sum()), 1e-12)

    rows: list[dict[str, Any]] = []
    for fam in families:
        part = used[used["family"] == fam]
        share_clients = float(len(part) / totals_clients)
        share_volume = float(part["total_tnd"].sum() / totals_volume)
        envelope_vals = [priors[p][fam] for p in ["nominal", "firm_count", "trade_flow"]]
        lo, hi = min(envelope_vals), max(envelope_vals)
        inside = (lo <= share_clients <= hi) or (lo <= share_volume <= hi)
        rows.append(
            {
                "family": fam,
                "client_count": int(len(part)),
                "total_tnd": float(part["total_tnd"].sum()),
                "share_clients": share_clients,
                "share_volume": share_volume,
                "nominal_prior_share": float(priors["nominal"][fam]),
                "firm_count_prior_share": float(priors["firm_count"][fam]),
                "trade_flow_prior_share": float(priors["trade_flow"][fam]),
                "inside_prior_envelope": bool(inside),
            }
        )
    out = pd.DataFrame(rows)
    nonempty_ok = bool((out["client_count"] > 0).all())
    inside_count = int(out["inside_prior_envelope"].sum())
    passed = nonempty_ok and inside_count >= 3
    return {
        "summary": {
            "verdict": "pass" if passed else "fail",
            "all_families_nonempty": nonempty_ok,
            "families_inside_envelope": inside_count,
        },
        "output": out,
    }


def _choose_best_scenario(
    strategy_ranking: pd.DataFrame,
    strategy_ranking_cols: dict[str, str],
    currency_pair: str,
    tenor_months: int,
    side: str,
) -> str | None:
    sr = strategy_ranking.copy()
    rank_col = strategy_ranking_cols["rank"]
    cp_col = strategy_ranking_cols["currency_pair"]
    tenor_col = strategy_ranking_cols["tenor_months"]
    side_col = strategy_ranking_cols["side"]
    stress_col = strategy_ranking_cols["forward_stress_scenario"]
    scenario_col = strategy_ranking_cols["hedge_intensity_scenario"]

    sr = sr[pd.to_numeric(sr[rank_col], errors="coerce") == 1]
    cip = sr[sr[stress_col].astype(str).str.strip().str.lower() == "cip_base"]
    if not cip.empty:
        sr = cip
    sr = sr[(sr[cp_col] == currency_pair) & (pd.to_numeric(sr[tenor_col], errors="coerce") == int(tenor_months))]
    sided = sr[sr[side_col] == side]
    if not sided.empty:
        sr = sided
    if sr.empty:
        return None
    counts = sr[scenario_col].astype(str).value_counts()
    return str(counts.index[0]) if len(counts) else None


def _test2_hedge_ratio_comparison(
    tx: pd.DataFrame,
    client_universe: pd.DataFrame,
    strategy_ranking: pd.DataFrame,
    strategy_ranking_cols: dict[str, str],
    hedge_recommendations: pd.DataFrame,
    hedge_recommendations_cols: dict[str, str],
) -> dict:
    """Compare observed hedge ratios vs model recommendations at client x currency x month level."""
    hdr_family_col = hedge_recommendations_cols["family"]
    hdr_cp_col = hedge_recommendations_cols["currency_pair"]
    hdr_tenor_col = hedge_recommendations_cols["tenor_months"]
    hdr_timing_col = hedge_recommendations_cols["timing_cv_scenario"]
    hdr_scenario_col = hedge_recommendations_cols["selected_hedge_intensity_scenario"]
    hdr_ratio_col = hedge_recommendations_cols["recommended_hedge_ratio"]

    used_clients = client_universe[client_universe["included_in_tests"] & client_universe["family"].isin(["importer", "exporter", "processor", "trader"])]
    tx_used = tx.merge(used_clients[["client", "family"]], left_on="Client", right_on="client", how="inner")
    tx_used["month_bucket"] = tx_used["Transaction date"].dt.to_period("M").dt.to_timestamp()
    tx_used["is_hedge"] = tx_used["instrument_class"].eq("hedge")
    tx_used["tenor_months_obs"] = tx_used["tenor_days"].astype(float) / 30.4

    rows: list[dict[str, Any]] = []
    for keys, g in tx_used.groupby(["client", "family", "currency_pair", "month_bucket"], sort=False):
        client, family, currency_pair, month_bucket = keys
        all_amount = float(g["Amount_TND"].sum())
        if all_amount <= 0:
            continue
        hedge_amount = float(g.loc[g["is_hedge"], "Amount_TND"].sum())
        h_observed = hedge_amount / all_amount
        hedge_tenor = g.loc[g["is_hedge"], "tenor_months_obs"]
        tenor_obs = float(pd.to_numeric(hedge_tenor, errors="coerce").median()) if len(hedge_tenor) else np.nan
        nearest_tenor = _snap_tenor_months(tenor_obs, default=3)
        side_amounts = g.groupby("side", sort=False)["Amount_TND"].sum()
        dominant_side = str(side_amounts.idxmax()) if len(side_amounts) else "ASK"
        best_scenario = _choose_best_scenario(
            strategy_ranking,
            strategy_ranking_cols,
            currency_pair,
            nearest_tenor,
            dominant_side,
        )

        lookup_status = "matched"
        n_matching = 0
        h_recommended = np.nan
        if best_scenario is None:
            lookup_status = "no_recommendation"
        else:
            filt = hedge_recommendations[
                (hedge_recommendations[hdr_family_col] == family)
                & (hedge_recommendations[hdr_cp_col] == currency_pair)
                & (pd.to_numeric(hedge_recommendations[hdr_tenor_col], errors="coerce") == int(nearest_tenor))
                & (hedge_recommendations[hdr_timing_col].astype(str).str.strip().str.lower() == "baseline")
                & (hedge_recommendations[hdr_scenario_col] == best_scenario)
            ].copy()
            n_matching = int(len(filt))
            if n_matching > 0:
                h_recommended = float(pd.to_numeric(filt[hdr_ratio_col], errors="coerce").median())
            else:
                lookup_status = "no_match"

        gap = float(h_recommended - h_observed) if np.isfinite(h_recommended) else np.nan
        rows.append(
            {
                "client": client,
                "family": family,
                "currency_pair": currency_pair,
                "month_bucket": month_bucket,
                "all_amount_tnd": all_amount,
                "hedge_amount_tnd": hedge_amount,
                "h_observed": h_observed,
                "nearest_model_tenor": int(nearest_tenor),
                "best_scenario": best_scenario,
                "h_recommended": h_recommended,
                "gap": gap,
                "abs_gap": abs(gap) if np.isfinite(gap) else np.nan,
                "within_pm20": bool(abs(gap) <= 0.20) if np.isfinite(gap) else False,
                "lookup_status": lookup_status,
                "n_matching_profiles": n_matching,
            }
        )

    detail = pd.DataFrame(rows)
    valid = detail[np.isfinite(detail["gap"])].copy()
    if valid.empty:
        summary = pd.DataFrame(
            columns=[
                "family",
                "currency_pair",
                "n_cells",
                "mean_gap",
                "median_gap",
                "std_gap",
                "median_abs_gap",
                "share_within_pm20",
                "share_model_higher",
            ]
        )
        verdict = False
        mean_gap = np.nan
        med_abs = np.nan
        share_pm20 = np.nan
    else:
        summary = (
            valid.groupby(["family", "currency_pair"], sort=False)
            .agg(
                n_cells=("gap", "size"),
                mean_gap=("gap", "mean"),
                median_gap=("gap", "median"),
                std_gap=("gap", "std"),
                median_abs_gap=("abs_gap", "median"),
                share_within_pm20=("within_pm20", "mean"),
                share_model_higher=("gap", lambda s: float((s > 0).mean())),
            )
            .reset_index()
        )
        mean_gap = float(valid["gap"].mean())
        med_abs = float(valid["abs_gap"].median())
        share_pm20 = float(valid["within_pm20"].mean())
        verdict = (med_abs <= 0.25) and (share_pm20 >= 0.50) and (abs(mean_gap) <= 0.10)

    return {
        "summary": {
            "verdict": "pass" if verdict else "fail",
            "median_abs_gap": med_abs,
            "share_within_pm20": share_pm20,
            "mean_gap": mean_gap,
            "n_cells_with_recommendation": int(len(valid)),
        },
        "detail": detail,
        "aggregate": summary,
    }


def _median_spread_table(tx: pd.DataFrame) -> pd.DataFrame:
    tmp = tx.copy()
    tmp["spread"] = pd.to_numeric(tmp["Rate"], errors="coerce") - pd.to_numeric(tmp["Mid market"], errors="coerce")
    out = (
        tmp.groupby(["currency_pair", "instrument_class"], sort=False)["spread"]
        .median()
        .rename("cell_median_spread")
        .reset_index()
    )
    return out


def _resolve_hc_for_deal(
    family: str,
    currency_pair: str,
    tenor_months: int,
    side: str,
    strategy_ranking: pd.DataFrame,
    strategy_ranking_cols: dict[str, str],
    hedge_recommendations: pd.DataFrame,
    hedge_recommendations_cols: dict[str, str],
) -> tuple[float, str]:
    best = _choose_best_scenario(strategy_ranking, strategy_ranking_cols, currency_pair, tenor_months, side)
    if best is None:
        return np.nan, "no_recommendation"
    hdr_family_col = hedge_recommendations_cols["family"]
    hdr_cp_col = hedge_recommendations_cols["currency_pair"]
    hdr_tenor_col = hedge_recommendations_cols["tenor_months"]
    hdr_timing_col = hedge_recommendations_cols["timing_cv_scenario"]
    hdr_scenario_col = hedge_recommendations_cols["selected_hedge_intensity_scenario"]
    hdr_ratio_col = hedge_recommendations_cols["recommended_hedge_ratio"]
    filt = hedge_recommendations[
        (hedge_recommendations[hdr_family_col] == family)
        & (hedge_recommendations[hdr_cp_col] == currency_pair)
        & (pd.to_numeric(hedge_recommendations[hdr_tenor_col], errors="coerce") == int(tenor_months))
        & (hedge_recommendations[hdr_timing_col].astype(str).str.strip().str.lower() == "baseline")
        & (hedge_recommendations[hdr_scenario_col] == best)
    ].copy()
    if filt.empty:
        return np.nan, "no_match"
    return float(pd.to_numeric(filt[hdr_ratio_col], errors="coerce").median()), "matched"


def _carry_proxy_from_summary(
    rolling_summary: pd.DataFrame,
    rolling_summary_cols: dict[str, str],
    currency_pair: str,
    tenor_months: int,
    side: str,
) -> float:
    cp_col = rolling_summary_cols["currency_pair"]
    tenor_col = rolling_summary_cols["tenor_months"]
    side_col = rolling_summary_cols["side"]
    scenario_col = rolling_summary_cols["hedge_intensity_scenario"]
    stress_col = rolling_summary_cols["forward_stress_scenario"]
    carry_col = rolling_summary_cols["mean_forward_advantage"]

    rs = rolling_summary.copy()
    rs = rs[(rs[cp_col] == currency_pair) & (pd.to_numeric(rs[tenor_col], errors="coerce") == int(tenor_months))]
    s = rs[rs[side_col] == side]
    if not s.empty:
        rs = s
    b = rs[rs[scenario_col].astype(str).str.strip().str.lower() == "baseline_protection"]
    if not b.empty:
        rs = b
    c = rs[rs[stress_col].astype(str).str.strip().str.lower() == "cip_base"]
    if not c.empty:
        rs = c
    if rs.empty:
        return 0.0
    raw = pd.to_numeric(rs[carry_col], errors="coerce").dropna()
    if raw.empty:
        return 0.0
    proxy = float(raw.mean())
    if str(carry_col).strip().lower().endswith("_bps"):
        return float(proxy / 10000.0)
    if abs(proxy) > 0.5:
        print(
            f"[validation] Warning: {carry_col} mean absolute value is {abs(proxy):.4f}; "
            "treating as fraction per specification."
        )
    return proxy


def _test3_counterfactual_pnl(
    tx: pd.DataFrame,
    client_universe: pd.DataFrame,
    spot_history: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    strategy_ranking: pd.DataFrame,
    strategy_ranking_cols: dict[str, str],
    hedge_recommendations: pd.DataFrame,
    hedge_recommendations_cols: dict[str, str],
    rolling_summary_cols: dict[str, str],
) -> dict:
    """
    Test 3: counterfactual P&L.

    Reports two improvement metrics per deal:

      improvement_mid          - uses Reuters mid as the counterfactual
                                  reference; matches the model's
                                  CIP-calibrated forward pricing.
                                  Upper bound on the model's value-add
                                  in a frictionless execution world.

      improvement_spread_adj   - adds the cell-median bank spread
                                  observed in the Colombus data back
                                  into the counterfactual, so the
                                  comparison is all-in vs all-in.
                                  Realistic floor on value-add.

    Test 3 pass criterion evaluates improvement_spread_adj only.
    """
    used_clients = client_universe[client_universe["included_in_tests"] & client_universe["family"].isin(["importer", "exporter", "processor", "trader"])]
    data = tx.merge(used_clients[["client", "family"]], left_on="Client", right_on="client", how="inner")
    lookup_spot = _make_realized_spot_lookup(spot_history)
    ib_min_date = pd.to_datetime(spot_history["date"], errors="coerce").min()
    spread_tbl = _median_spread_table(data)
    spread_map = {
        (r["currency_pair"], r["instrument_class"]): float(r["cell_median_spread"])
        for _, r in spread_tbl.iterrows()
    }

    rows: list[dict[str, Any]] = []
    for _, r in data.iterrows():
        pair = str(r["currency_pair"])
        side = str(r["side"])
        is_buy = str(r["Type"]).strip().lower() == "buy"
        tenor_months = 1 if r["instrument_class"] == "spot" else _snap_tenor_months(float(r["tenor_days"]) / 30.4, default=3)
        h_c, status = _resolve_hc_for_deal(
            family=str(r["family"]),
            currency_pair=pair,
            tenor_months=tenor_months,
            side=side,
            strategy_ranking=strategy_ranking,
            strategy_ranking_cols=strategy_ranking_cols,
            hedge_recommendations=hedge_recommendations,
            hedge_recommendations_cols=hedge_recommendations_cols,
        )
        if not np.isfinite(h_c):
            continue

        amount = float(r["Amount"])
        r_exec = float(r["Rate"])
        tx_date = pd.to_datetime(r["Transaction date"])
        val_date = pd.to_datetime(r["Value date"])
        realized_date = tx_date if r["instrument_class"] == "spot" else val_date
        realized_spot = lookup_spot(realized_date, pair)
        if r["instrument_class"] == "spot":
            carry = _carry_proxy_from_summary(rolling_summary, rolling_summary_cols, pair, tenor_months, side)
            try:
                spot_m1 = lookup_spot(tx_date - pd.Timedelta(days=30), pair)
            except ValueError as exc:
                if "before first IB observation" in str(exc) and not pd.isna(ib_min_date):
                    spot_m1 = lookup_spot(ib_min_date, pair)
                else:
                    raise
            f_synth = spot_m1 * (1.0 + carry)
            cf_counter_mid = (h_c * f_synth + (1.0 - h_c) * realized_spot) * amount
        else:
            f_synth = np.nan
            cf_counter_mid = (h_c * r_exec + (1.0 - h_c) * realized_spot) * amount

        cf_actual = r_exec * amount
        spread = spread_map.get((pair, str(r["instrument_class"])), 0.0)
        a_sign = 1.0 if is_buy else -1.0
        cf_counter_spread_adj = cf_counter_mid + spread * amount * a_sign
        denom = max(amount * realized_spot, 1e-12)
        if is_buy:
            improvement_mid = (cf_actual - cf_counter_mid) / denom
            improvement_spread_adj = (cf_actual - cf_counter_spread_adj) / denom
            side_label = "outflow"
        else:
            improvement_mid = (cf_counter_mid - cf_actual) / denom
            improvement_spread_adj = (cf_counter_spread_adj - cf_actual) / denom
            side_label = "inflow"

        rows.append(
            {
                "client": r["client"],
                "family": r["family"],
                "date": tx_date,
                "currency": r["Currency"],
                "currency_pair": pair,
                "type": r["Type"],
                "instrument": r["Instrument"],
                "instrument_class": r["instrument_class"],
                "side": side_label,
                "amount": amount,
                "R_exec": r_exec,
                "F_synth": f_synth,
                "realized_spot": realized_spot,
                "h_c_recommended": h_c,
                "lookup_status": status,
                "CF_actual": cf_actual,
                "CF_counter_mid": cf_counter_mid,
                "CF_counter_spread_adj": cf_counter_spread_adj,
                "improvement_mid": improvement_mid,
                "improvement_spread_adj": improvement_spread_adj,
            }
        )

    detail = pd.DataFrame(rows)
    if detail.empty:
        summary = pd.DataFrame(
            columns=[
                "family",
                "currency_pair",
                "side",
                "n_deals",
                "mean_improvement_mid",
                "mean_improvement_spread_adj",
                "std_improvement_mid",
                "std_improvement_spread_adj",
                "share_positive_mid",
                "share_positive_spread_adj",
                "variance_reduction",
            ]
        )
        verdict = False
        weighted_mean = np.nan
        share_cells_pos_var = np.nan
    else:
        summary = (
            detail.groupby(["family", "currency_pair", "side"], sort=False)
            .agg(
                n_deals=("improvement_spread_adj", "size"),
                mean_improvement_mid=("improvement_mid", "mean"),
                mean_improvement_spread_adj=("improvement_spread_adj", "mean"),
                std_improvement_mid=("improvement_mid", "std"),
                std_improvement_spread_adj=("improvement_spread_adj", "std"),
                share_positive_mid=("improvement_mid", lambda s: float((s > 0).mean())),
                share_positive_spread_adj=("improvement_spread_adj", lambda s: float((s > 0).mean())),
                std_cf_actual=("CF_actual", "std"),
                std_cf_counter_mid=("CF_counter_mid", "std"),
                std_cf_counter_spread_adj=("CF_counter_spread_adj", "std"),
            )
            .reset_index()
        )
        summary["variance_reduction"] = 1.0 - (
            summary["std_cf_counter_spread_adj"] / summary["std_cf_actual"].replace(0, np.nan)
        )
        weights = summary["n_deals"].replace(0, np.nan)
        weighted_mean = float(np.average(summary["mean_improvement_spread_adj"], weights=weights))
        share_cells_pos_var = float((summary["variance_reduction"] > 0).mean())
        verdict = (weighted_mean > 0) or (share_cells_pos_var >= 0.60)

    return {
        "summary": {
            "verdict": "pass" if verdict else "fail",
            "weighted_mean_improvement_spread_adj": weighted_mean,
            "share_cells_positive_variance_reduction": share_cells_pos_var,
            "n_deals_evaluated": int(len(detail)),
        },
        "detail": detail,
        "aggregate": summary,
    }


def _test4_tenor_distribution(
    tx: pd.DataFrame,
    client_universe: pd.DataFrame,
    stage15_handoff: pd.DataFrame,
    stage15_handoff_cols: dict[str, str],
) -> dict:
    """Compare empirical client tenor distributions against model tenor allocations."""
    used = client_universe[client_universe["included_in_tests"] & client_universe["family"].isin(["importer", "exporter", "processor", "trader"])].copy()
    data = tx.merge(used[["client", "family", "buy_share", "sell_share", "n_hedge_deals"]], left_on="Client", right_on="client", how="inner")
    data = data[data["instrument_class"] == "hedge"].copy()
    data["tenor_months_emp"] = data["tenor_days"].astype(float) / 30.4
    data["tenor_bucket"] = data["tenor_months_emp"].apply(lambda x: _snap_tenor_months(x, default=3))

    family_col = stage15_handoff_cols["family"]
    c_col = stage15_handoff_cols["c"]
    omega_col = stage15_handoff_cols["omega_t"]
    tenor_col = stage15_handoff_cols["tenor_months"]
    direction_col = stage15_handoff_cols["direction"]

    hand = stage15_handoff.copy()
    hand[c_col] = pd.to_numeric(hand[c_col], errors="coerce")
    hand[omega_col] = pd.to_numeric(hand[omega_col], errors="coerce")
    hand[tenor_col] = pd.to_numeric(hand[tenor_col], errors="coerce")
    family_median_c = hand.groupby(family_col, sort=False)[c_col].median().to_dict()

    detail_rows: list[dict[str, Any]] = []
    client_rows: list[dict[str, Any]] = []
    for client, g in data.groupby("client", sort=False):
        n_hedge_deals = int(len(g))
        if n_hedge_deals < 5:
            continue
        family = str(g["family"].iloc[0])
        buy_share = float(g["buy_share"].iloc[0])
        sell_share = float(g["sell_share"].iloc[0])
        if sell_share >= 0.70:
            direction_filter = "inflow"
        elif buy_share >= 0.70:
            direction_filter = "outflow"
        else:
            direction_filter = "both"

        fam_c = float(family_median_c.get(family, np.nan))
        model_pool = hand[(hand[family_col] == family) & hand[c_col].notna() & (hand[tenor_col].isin(TENOR_BUCKETS))]
        if np.isfinite(fam_c):
            model_pool = model_pool[(model_pool[c_col] - fam_c).abs() <= 0.5]
        if direction_filter != "both":
            model_pool = model_pool[model_pool[direction_col] == direction_filter]
        if model_pool.empty:
            continue

        model_w = model_pool.groupby(tenor_col, sort=False)[omega_col].sum()
        model_w = model_w.reindex(TENOR_BUCKETS, fill_value=0.0)
        model_w = model_w / max(float(model_w.sum()), 1e-12)

        emp_w = g.groupby("tenor_bucket", sort=False)["Amount_TND"].sum()
        emp_w = emp_w.reindex(TENOR_BUCKETS, fill_value=0.0)
        emp_w = emp_w / max(float(emp_w.sum()), 1e-12)

        l1 = 0.5 * float(np.abs(emp_w.values - model_w.values).sum())
        emp_mean_tenor = float((emp_w.index.to_numpy(dtype=float) * emp_w.values).sum())
        model_mean_tenor = float((model_w.index.to_numpy(dtype=float) * model_w.values).sum())

        for t in TENOR_BUCKETS:
            detail_rows.append(
                {
                    "client": client,
                    "family": family,
                    "direction_filter": direction_filter,
                    "tenor_months": int(t),
                    "empirical_weight": float(emp_w.loc[t]),
                    "model_weight": float(model_w.loc[t]),
                    "abs_gap": float(abs(emp_w.loc[t] - model_w.loc[t])),
                    "n_hedge_deals": n_hedge_deals,
                }
            )

        client_rows.append(
            {
                "client": client,
                "family": family,
                "n_hedge_deals": n_hedge_deals,
                "l1_distance": l1,
                "empirical_mean_tenor": emp_mean_tenor,
                "model_mean_tenor": model_mean_tenor,
                "row_type": "client",
            }
        )

    detail = pd.DataFrame(detail_rows)
    client_summary = pd.DataFrame(client_rows)
    if client_summary.empty:
        family_agg = pd.DataFrame(columns=["client", "family", "n_hedge_deals", "l1_distance", "empirical_mean_tenor", "model_mean_tenor", "row_type"])
        tenor_summary = client_summary.copy()
        verdict = False
        med_l1 = np.nan
        emp_mean_global = np.nan
    else:
        family_agg = (
            client_summary.groupby("family", sort=False)
            .agg(
                n_hedge_deals=("n_hedge_deals", "sum"),
                l1_distance=("l1_distance", "median"),
                empirical_mean_tenor=("empirical_mean_tenor", "mean"),
                model_mean_tenor=("model_mean_tenor", "mean"),
            )
            .reset_index()
        )
        family_agg.insert(0, "client", "__family__")
        family_agg["row_type"] = "family_aggregate"
        tenor_summary = pd.concat([client_summary, family_agg], ignore_index=True)
        med_l1 = float(client_summary["l1_distance"].median())
        emp_mean_global = float(client_summary["empirical_mean_tenor"].mean())
        verdict = (med_l1 <= 0.40) and (1.0 <= emp_mean_global <= 12.0)

    return {
        "summary": {
            "verdict": "pass" if verdict else "fail",
            "median_l1_distance": med_l1,
            "empirical_mean_tenor_global": emp_mean_global,
            "n_clients_scored": int(len(client_summary)),
        },
        "detail": detail,
        "aggregate": tenor_summary,
    }


def _require_latest_outputs(outputs_dir: Path) -> dict[str, pd.DataFrame]:
    required = [
        "Family_Profile_Summary.csv",
        "Stage_1_5_Handoff.csv",
        "Rolling_Market_Summary.csv",
        "Hedge_Decision_Recommendations.csv",
        "Strategy_Ranking.csv",
    ]
    loaded: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for name in required:
        p = outputs_dir / name
        if not p.exists():
            missing.append(name)
            continue
        loaded[name] = pd.read_csv(p)
    if missing:
        raise FileNotFoundError(
            "Missing required pipeline outputs in data/outputs/latest. "
            f"Missing: {missing}. Run pipeline_full.py first."
        )
    return loaded


def run_external_validation(
    colombus_path: Path,
    outputs_dir: Path,
    *,
    min_deals_per_client: int = 10,
    family_thresholds: dict[str, float] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the four validation tests and write CSVs to outputs_dir.

    Parameters
    ----------
    colombus_path : Path
        Path to the colombus_transactions.xlsx file.
    outputs_dir : Path
        Path to data/outputs/latest/.
    min_deals_per_client : int, default 10
        Clients with fewer deals are excluded from family classification
        and hedge-ratio comparison.
    family_thresholds : dict[str, float] | None
        Optional override of thresholds.
    verbose : bool, default True
        If True, print progress per test.

    Returns
    -------
    dict[str, Any]
        Dict with keys test1, test2, test3, test4.
    """
    thresholds = dict(DEFAULT_THRESHOLDS if family_thresholds is None else family_thresholds)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    _log("Loading Colombus workbook...", verbose)
    tx, ib = _load_colombus(colombus_path)

    _log("Loading existing pipeline outputs...", verbose)
    refs = _require_latest_outputs(outputs_dir)
    family_summary = refs["Family_Profile_Summary.csv"]
    stage15_handoff = refs["Stage_1_5_Handoff.csv"]
    rolling_summary = refs["Rolling_Market_Summary.csv"]
    hedge_recommendations = refs["Hedge_Decision_Recommendations.csv"]
    strategy_ranking = refs["Strategy_Ranking.csv"]
    strategy_ranking_cols = _resolve_columns(strategy_ranking, STRATEGY_RANKING_CANDIDATES, "Strategy_Ranking.csv")
    hedge_recommendations_cols = _resolve_columns(
        hedge_recommendations,
        HEDGE_RECOMMENDATIONS_CANDIDATES,
        "Hedge_Decision_Recommendations.csv",
    )
    stage15_handoff_cols = _resolve_columns(stage15_handoff, STAGE15_HANDOFF_CANDIDATES, "Stage_1_5_Handoff.csv")
    rolling_summary_cols = _resolve_columns(rolling_summary, ROLLING_SUMMARY_CANDIDATES, "Rolling_Market_Summary.csv")

    _log("Building client universe...", verbose)
    universe = _classify_clients(tx, min_deals_per_client, thresholds)
    universe = universe[CLIENT_UNIVERSE_COLUMNS].copy()
    universe.to_csv(outputs_dir / "Validation_Client_Universe.csv", index=False)

    _log("Running Test 1 (family distribution)...", verbose)
    t1 = _test1_family_distribution(universe, family_summary)
    t1["output"].to_csv(outputs_dir / "Validation_Family_Distribution.csv", index=False)

    _log("Running Test 2 (hedge ratio comparison)...", verbose)
    t2 = _test2_hedge_ratio_comparison(
        tx,
        universe,
        strategy_ranking,
        strategy_ranking_cols,
        hedge_recommendations,
        hedge_recommendations_cols,
    )
    t2["detail"].to_csv(outputs_dir / "Validation_HedgeRatio_Comparison.csv", index=False)
    t2["aggregate"].to_csv(outputs_dir / "Validation_HedgeRatio_Summary.csv", index=False)

    _log("Running Test 3 (counterfactual P&L)...", verbose)
    t3 = _test3_counterfactual_pnl(
        tx,
        universe,
        ib,
        rolling_summary,
        strategy_ranking,
        strategy_ranking_cols,
        hedge_recommendations,
        hedge_recommendations_cols,
        rolling_summary_cols,
    )
    t3["detail"].to_csv(outputs_dir / "Validation_Counterfactual_PnL.csv", index=False)
    t3["aggregate"].to_csv(outputs_dir / "Validation_Counterfactual_Summary.csv", index=False)

    _log("Running Test 4 (tenor distribution)...", verbose)
    t4 = _test4_tenor_distribution(tx, universe, stage15_handoff, stage15_handoff_cols)
    t4["detail"].to_csv(outputs_dir / "Validation_Tenor_Distribution.csv", index=False)
    t4["aggregate"].to_csv(outputs_dir / "Validation_Tenor_Summary.csv", index=False)

    headline = pd.DataFrame(
        [
            {
                "test1_pass": t1["summary"]["verdict"] == "pass",
                "test1_metric": t1["summary"]["families_inside_envelope"],
                "test2_pass": t2["summary"]["verdict"] == "pass",
                "test2_metric": t2["summary"]["median_abs_gap"],
                "test3_pass": t3["summary"]["verdict"] == "pass",
                "test3_metric": t3["summary"]["weighted_mean_improvement_spread_adj"],
                "test4_pass": t4["summary"]["verdict"] == "pass",
                "test4_metric": t4["summary"]["median_l1_distance"],
                "n_clients": int(universe["client"].nunique()),
                "n_deals": int(len(tx)),
                "date_range_start": pd.to_datetime(tx["Transaction date"]).min(),
                "date_range_end": pd.to_datetime(tx["Transaction date"]).max(),
                "run_timestamp": pd.Timestamp.utcnow(),
            }
        ]
    )
    headline.to_csv(outputs_dir / "Validation_Headline_Metrics.csv", index=False)

    _log("External validation outputs written.", verbose)
    return {
        "test1": t1["summary"],
        "test2": t2["summary"],
        "test3": t3["summary"],
        "test4": t4["summary"],
    }
