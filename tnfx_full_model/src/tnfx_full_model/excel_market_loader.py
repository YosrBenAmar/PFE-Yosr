from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import MARKET_COLUMNS


SPOT_HISTORY_COLUMNS = ["date", "currency_pair", "spot_bid", "spot_ask", "spot_mid"]

ALIASES = {
    "valuation_date": ["valuation_date", "as_of_date", "value_date", "pricing_date"],
    "date": ["date", "exchange_date", "history_date", "spot_date", "observation_date"],
    "currency_pair": ["currency_pair", "ccy_pair", "fx_pair", "pair", "currency", "pair_name"],
    "tenor_months": ["tenor_months", "tenor_month", "tenor_m", "months", "maturity_months", "tenor"],
    "tenor_days": ["tenor_days", "days", "maturity_days", "day_count_days"],
    "spot_bid": ["spot_bid", "bid_spot", "fx_spot_bid", "bid"],
    "spot_ask": ["spot_ask", "ask_spot", "fx_spot_ask", "ask"],
    "spot_mid": ["spot_mid", "mid_spot", "mid_price", "spot", "spot_rate", "fx_spot", "fx_rate"],
    "tnd_rate_bid": ["tnd_rate_bid", "tnd_bid", "domestic_rate_bid", "domestic_bid", "tnd_yield_bid"],
    "tnd_rate_ask": ["tnd_rate_ask", "tnd_ask", "domestic_rate_ask", "domestic_ask", "tnd_yield_ask"],
    "tnd_rate_mid": ["tnd_rate_mid", "tnd_mid", "domestic_rate", "domestic_rate_mid", "tnd_yield"],
    "fcy_rate_bid": ["fcy_rate_bid", "foreign_rate_bid", "foreign_bid", "eur_usd_rate_bid", "fcy_yield_bid"],
    "fcy_rate_ask": ["fcy_rate_ask", "foreign_rate_ask", "foreign_ask", "eur_usd_rate_ask", "fcy_yield_ask"],
    "fcy_rate_mid": ["fcy_rate_mid", "foreign_rate", "foreign_rate_mid", "fcy_yield"],
    "forward_bid": ["forward_bid", "fwd_bid", "outright_forward_bid"],
    "forward_ask": ["forward_ask", "fwd_ask", "outright_forward_ask"],
    "forward_mid": ["forward_mid", "fwd_mid", "forward", "fwd", "outright_forward"],
    "realized_future_date": ["realized_future_date", "future_date", "settlement_date", "realized_date"],
    "realized_future_spot_bid": ["realized_future_spot_bid", "future_spot_bid", "realized_spot_bid"],
    "realized_future_spot_ask": ["realized_future_spot_ask", "future_spot_ask", "realized_spot_ask"],
    "realized_future_spot_mid": ["realized_future_spot_mid", "future_spot_mid", "realized_spot", "realized_spot_mid"],
}


def normalize_header(value) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_header(c) for c in out.columns]
    rename = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            if alias in out.columns and canonical not in out.columns:
                rename[alias] = canonical
                break
    return out.rename(columns=rename)


def _infer_currency_pair(sheet_name: str) -> str | None:
    name = normalize_header(sheet_name)
    if "eur" in name and ("tnd" in name or "historical_spot" in name or "forward" in name):
        return "EUR_TND"
    if "usd" in name and ("tnd" in name or "historical_spot" in name or "forward" in name):
        return "USD_TND"
    return None


def _clean_pair(value) -> str:
    text = str(value).upper().replace("/", "_").replace("-", "_").replace(" ", "_")
    if "EUR" in text and "TND" in text:
        return "EUR_TND"
    if "USD" in text and "TND" in text:
        return "USD_TND"
    if text == "EUR":
        return "EUR_TND"
    if text == "USD":
        return "USD_TND"
    return text


def _coerce_tenor_months(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        match = re.search(r"(\d+(?:\.\d+)?)", value)
        if not match:
            return np.nan
        return float(match.group(1))
    return float(value)


def _read_all_sheets(path: Path) -> dict[str, pd.DataFrame]:
    try:
        return pd.read_excel(path, sheet_name=None, header=None)
    except Exception as exc:
        raise ValueError(f"Could not read market workbook {path}: {exc}") from exc


def _detect_header_frame(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.empty:
        return raw
    alias_tokens = {alias for aliases in ALIASES.values() for alias in aliases}
    alias_tokens.update({"transaction_date", "currency", "historical_spot", "1m_hedge_forward_rate"})
    best_idx = raw.index[0]
    best_score = -1
    for idx, row in raw.head(40).iterrows():
        normalized = [normalize_header(v) for v in row.tolist() if not pd.isna(v)]
        score = sum(1 for value in normalized if value in alias_tokens or "hedge_forward_rate" in value or "hedge_spot_rate" in value)
        if score > best_score:
            best_idx = idx
            best_score = score
    header = [normalize_header(v) if not pd.isna(v) else f"unnamed_{i}" for i, v in enumerate(raw.loc[best_idx].tolist())]
    data = raw.loc[raw.index > best_idx].copy()
    data.columns = header
    return data.dropna(how="all")


def _excel_serial_to_datetime(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    parsed = pd.to_datetime(series, errors="coerce")
    mask = numeric.notna()
    parsed.loc[mask] = pd.to_datetime(numeric.loc[mask], unit="D", origin="1899-12-30")
    return parsed


def _prepare_sheet(sheet_name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_headers(_detect_header_frame(df))
    if df.empty:
        return df
    inferred = _infer_currency_pair(sheet_name)
    if inferred and "currency_pair" not in df.columns:
        df["currency_pair"] = inferred
    if "currency_pair" in df.columns:
        df["currency_pair"] = df["currency_pair"].map(_clean_pair)
    if "tenor_months" in df.columns:
        df["tenor_months"] = df["tenor_months"].map(_coerce_tenor_months)
    if "tenor_days" not in df.columns and "tenor_months" in df.columns:
        df["tenor_days"] = (df["tenor_months"].astype(float) * 30).round().astype("Int64")
    return df


def _tenor_column_candidates(tenor: int, suffix: str) -> list[str]:
    labels = [f"{tenor}m"]
    if tenor == 12:
        labels.extend(["1y", "12m"])
    return [f"{label}_hedge_{suffix}" for label in labels]


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _parse_forward_strategy_sheets(sheets: dict[str, pd.DataFrame], valuation_date: str) -> tuple[pd.DataFrame, list[str]] | None:
    parsed: dict[tuple[str, int], dict] = {}
    used_sheets = []
    for name, raw in sheets.items():
        lowered = normalize_header(name)
        if "forward" not in lowered or ("ask" not in lowered and "bid" not in lowered):
            continue
        currency = "EUR" if "eur" in lowered else "USD" if "usd" in lowered else None
        side = "ask" if "ask" in lowered else "bid"
        if currency is None:
            continue
        df = _prepare_sheet(name, raw)
        if df.empty:
            continue
        used_sheets.append(name)
        for tenor in [1, 2, 3, 6, 9, 12]:
            days_col = _first_existing(df, _tenor_column_candidates(tenor, "n_of_days"))
            spot_col = _first_existing(df, _tenor_column_candidates(tenor, "spot_rate"))
            domestic_col = _first_existing(df, _tenor_column_candidates(tenor, "domestic_yield"))
            foreign_col = _first_existing(df, _tenor_column_candidates(tenor, "foreign_yield"))
            forward_col = _first_existing(df, _tenor_column_candidates(tenor, "forward_rate"))
            if not all([days_col, spot_col, domestic_col, foreign_col, forward_col]):
                continue
            cols = [days_col, spot_col, domestic_col, foreign_col, forward_col]
            tmp = df[cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).dropna(subset=cols)
            if tmp.empty:
                continue
            row = tmp.iloc[0]
            key = (f"{currency}_TND", tenor)
            out = parsed.setdefault(key, {
                "valuation_date": valuation_date,
                "currency_pair": f"{currency}_TND",
                "tenor_months": tenor,
                "tenor_days": int(round(float(row[days_col]))),
                "forward_bid": np.nan,
                "forward_ask": np.nan,
                "realized_future_date": np.nan,
                "realized_future_spot_bid": np.nan,
                "realized_future_spot_ask": np.nan,
                "source_quality_flag": "observed_bid_ask",
            })
            if side == "ask":
                out["spot_ask"] = float(row[spot_col])
                out["tnd_rate_ask"] = float(row[domestic_col])
                out["fcy_rate_bid"] = float(row[foreign_col])
                out["forward_ask"] = float(row[forward_col])
            else:
                out["spot_bid"] = float(row[spot_col])
                out["tnd_rate_bid"] = float(row[domestic_col])
                out["fcy_rate_ask"] = float(row[foreign_col])
                out["forward_bid"] = float(row[forward_col])
    if not parsed:
        return None
    rows = []
    for _, row in parsed.items():
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows), used_sheets


def _extract_tenor_rates_from_yields(sheets: dict[str, pd.DataFrame], side: str) -> dict[tuple[str, int], float]:
    target_sheet = "yields_ask" if side == "ask" else "yields_bid"
    out: dict[tuple[str, int], float] = {}
    for name, raw in sheets.items():
        if normalize_header(name) != target_sheet:
            continue
        df = _prepare_sheet(name, raw)
        if df.empty:
            continue
        mapping = {
            ("USD_TND", 1): "usd1md",
            ("USD_TND", 2): "usd2md",
            ("USD_TND", 3): "usd3md",
            ("USD_TND", 6): "usd6md",
            ("USD_TND", 9): "usd9md",
            ("USD_TND", 12): "usd1yd",
            ("EUR_TND", 1): "eur1md",
            ("EUR_TND", 2): "eur2md",
            ("EUR_TND", 3): "eur3md",
            ("EUR_TND", 6): "eur6md",
            ("EUR_TND", 9): "eur9md",
            ("EUR_TND", 12): "eur1yd",
        }
        for key, col in mapping.items():
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if not vals.empty:
                    out[key] = float(vals.iloc[0]) / 100.0
        break
    return out


def _extract_spot_bid_ask_from_history(sheets: dict[str, pd.DataFrame]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for name, raw in sheets.items():
        lowered = normalize_header(name)
        if "historical_spot" not in lowered:
            continue
        df = _prepare_sheet(name, raw)
        if df.empty:
            continue
        if "spot_bid" not in df.columns or "spot_ask" not in df.columns:
            continue
        pair = _infer_currency_pair(name)
        if pair is None:
            continue
        bids = pd.to_numeric(df["spot_bid"], errors="coerce").dropna()
        asks = pd.to_numeric(df["spot_ask"], errors="coerce").dropna()
        if not bids.empty and not asks.empty:
            out[pair] = (float(bids.iloc[0]), float(asks.iloc[0]))
    return out


def _enrich_forward_rows(rows: pd.DataFrame, sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if rows.empty:
        return rows
    rows = rows.copy()
    bid_rates = _extract_tenor_rates_from_yields(sheets, "bid")
    ask_rates = _extract_tenor_rates_from_yields(sheets, "ask")
    spots = _extract_spot_bid_ask_from_history(sheets)
    for idx, row in rows.iterrows():
        key = (row["currency_pair"], int(row["tenor_months"]))
        if ("spot_bid" not in rows.columns or pd.isna(row.get("spot_bid"))) and row["currency_pair"] in spots:
            rows.loc[idx, "spot_bid"] = spots[row["currency_pair"]][0]
        if ("spot_ask" not in rows.columns or pd.isna(row.get("spot_ask"))) and row["currency_pair"] in spots:
            rows.loc[idx, "spot_ask"] = spots[row["currency_pair"]][1]
        if pd.isna(row.get("tnd_rate_bid")) and key in bid_rates:
            rows.loc[idx, "tnd_rate_bid"] = bid_rates[key]
        if pd.isna(row.get("tnd_rate_ask")) and key in ask_rates:
            rows.loc[idx, "tnd_rate_ask"] = ask_rates[key]
        if pd.isna(row.get("fcy_rate_bid")) and key in bid_rates:
            rows.loc[idx, "fcy_rate_bid"] = bid_rates[key]
        if pd.isna(row.get("fcy_rate_ask")) and key in ask_rates:
            rows.loc[idx, "fcy_rate_ask"] = ask_rates[key]
    if "spot_mid" not in rows.columns:
        rows["spot_mid"] = np.nan
    spot_mid_missing = rows["spot_mid"].isna() & rows["spot_bid"].notna() & rows["spot_ask"].notna()
    rows.loc[spot_mid_missing, "spot_mid"] = (rows.loc[spot_mid_missing, "spot_bid"] + rows.loc[spot_mid_missing, "spot_ask"]) / 2.0
    for optional in ["forward_bid", "forward_ask", "realized_future_date", "realized_future_spot_bid", "realized_future_spot_ask"]:
        if optional not in rows.columns:
            rows[optional] = np.nan
    return rows


def _synth_spread(df: pd.DataFrame, mid_col: str, bid_col: str, ask_col: str, spread_bps: float, is_rate: bool) -> pd.DataFrame:
    if bid_col in df.columns and ask_col in df.columns:
        return df
    if mid_col not in df.columns:
        return df
    if is_rate:
        half = spread_bps / 20000.0
        df[bid_col] = df[mid_col] - half
        df[ask_col] = df[mid_col] + half
    else:
        half = spread_bps / 20000.0
        df[bid_col] = df[mid_col] * (1.0 - half)
        df[ask_col] = df[mid_col] * (1.0 + half)
    df["source_quality_flag"] = "synthetic_bid_ask"
    return df


def _complete_bid_ask(df: pd.DataFrame, config: dict, table_name: str) -> pd.DataFrame:
    df = df.copy()
    df["source_quality_flag"] = df.get("source_quality_flag", "observed_bid_ask")
    allow = bool(config.get("allow_bid_ask_synthesis", False))
    if allow:
        df = _synth_spread(df, "spot_mid", "spot_bid", "spot_ask", config.get("synthetic_fx_spread_bps", 20), False)
        df = _synth_spread(df, "tnd_rate_mid", "tnd_rate_bid", "tnd_rate_ask", config.get("synthetic_rate_spread_bps", 10), True)
        df = _synth_spread(df, "fcy_rate_mid", "fcy_rate_bid", "fcy_rate_ask", config.get("synthetic_rate_spread_bps", 10), True)
        df = _synth_spread(df, "forward_mid", "forward_bid", "forward_ask", config.get("synthetic_fx_spread_bps", 20), False)
        df = _synth_spread(df, "realized_future_spot_mid", "realized_future_spot_bid", "realized_future_spot_ask", config.get("synthetic_fx_spread_bps", 20), False)
    needed = ["spot_bid", "spot_ask"] if table_name == "spot_history" else ["spot_bid", "spot_ask", "tnd_rate_bid", "tnd_rate_ask", "fcy_rate_bid", "fcy_rate_ask"]
    missing_bid_ask = [c for c in needed if c not in df.columns]
    if missing_bid_ask:
        mid_cols = [c for c in ["spot_mid", "tnd_rate_mid", "fcy_rate_mid"] if c in df.columns]
        if mid_cols and not allow:
            raise ValueError(
                f"Workbook contains mid columns {mid_cols} but missing bid/ask columns {missing_bid_ask}. "
                "Set allow_bid_ask_synthesis: true to synthesize bid/ask explicitly."
            )
    if table_name == "market_data" and "forward_mid" in df.columns and not {"forward_bid", "forward_ask"}.issubset(df.columns) and not allow:
        raise ValueError("Workbook contains forward_mid but missing forward_bid/forward_ask. Set allow_bid_ask_synthesis: true to synthesize bid/ask explicitly.")
    return df


def _find_market_data(sheets: dict[str, pd.DataFrame], config: dict, valuation_date: str) -> tuple[pd.DataFrame, list[str]]:
    strategy = _parse_forward_strategy_sheets(sheets, valuation_date)
    if strategy is not None:
        rows, used = strategy
        rows = _enrich_forward_rows(rows, sheets)
        missing = [c for c in MARKET_COLUMNS if c not in rows.columns]
        if missing:
            raise ValueError(f"Workbook forward sheets parsed but market columns are still missing: {missing}")
        required = ["spot_bid", "spot_ask", "tnd_rate_bid", "tnd_rate_ask", "fcy_rate_bid", "fcy_rate_ask", "tenor_days"]
        missing_data = []
        for _, r in rows.iterrows():
            absent = [c for c in required if pd.isna(r[c])]
            if absent:
                missing_data.append({"currency_pair": r["currency_pair"], "tenor_months": int(r["tenor_months"]), "missing": absent})
        if missing_data:
            raise ValueError(f"Forward workbook sheets are missing bid/ask side data after yields/spot fallback: {missing_data[:5]}")
        return rows[MARKET_COLUMNS + [c for c in ["source_quality_flag"] if c in rows.columns]], used
    candidates = []
    for name, raw in sheets.items():
        df = _prepare_sheet(name, raw)
        if {"currency_pair", "tenor_months"}.issubset(df.columns):
            candidates.append((name, df))
    errors = []
    for name, df in candidates:
        try:
            df = _complete_bid_ask(df, config, "market_data")
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
            continue
        if "valuation_date" not in df.columns:
            df["valuation_date"] = valuation_date
        if "spot_mid" not in df.columns and {"spot_bid", "spot_ask"}.issubset(df.columns):
            df["spot_mid"] = (df["spot_bid"] + df["spot_ask"]) / 2.0
        for optional in ["forward_bid", "forward_ask", "realized_future_date", "realized_future_spot_bid", "realized_future_spot_ask"]:
            if optional not in df.columns:
                df[optional] = np.nan
        if "tenor_days" not in df.columns:
            df["tenor_days"] = (df["tenor_months"].astype(float) * 30).round().astype(int)
        missing = [c for c in MARKET_COLUMNS if c not in df.columns]
        if not missing:
            out = df[MARKET_COLUMNS + [c for c in ["source_quality_flag"] if c in df.columns]].copy()
            out["valuation_date"] = pd.to_datetime(out["valuation_date"]).dt.strftime("%Y-%m-%d")
            out = out[out["valuation_date"] == valuation_date]
            if not out.empty:
                return out, [name]
        else:
            errors.append(f"{name}: missing {missing}")
    raise ValueError("Workbook does not contain enough required market data. " + " | ".join(errors[:10]))


def _find_spot_history(sheets: dict[str, pd.DataFrame], config: dict) -> tuple[pd.DataFrame, list[str]]:
    errors = []
    frames = []
    used = []
    for name, raw in sheets.items():
        df = _prepare_sheet(name, raw)
        if not {"date", "currency_pair"}.issubset(df.columns):
            continue
        try:
            df = _complete_bid_ask(df, config, "spot_history")
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
            continue
        if "spot_mid" not in df.columns and {"spot_bid", "spot_ask"}.issubset(df.columns):
            df["spot_mid"] = (df["spot_bid"] + df["spot_ask"]) / 2.0
        missing = [c for c in SPOT_HISTORY_COLUMNS if c not in df.columns]
        if not missing:
            out = df[SPOT_HISTORY_COLUMNS + [c for c in ["source_quality_flag"] if c in df.columns]].copy()
            out["date"] = _excel_serial_to_datetime(out["date"])
            frames.append(out)
            used.append(name)
            continue
        errors.append(f"{name}: missing {missing}")
    if frames:
        return pd.concat(frames, ignore_index=True).sort_values(["currency_pair", "date"]), used
    raise ValueError("Workbook does not contain enough required spot history. " + " | ".join(errors[:10]))


def load_excel_market_workbook(path: str | Path, config: dict, valuation_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Market workbook not found: {path}")
    sheets = _read_all_sheets(path)
    market, market_sheets = _find_market_data(sheets, config, valuation_date)
    spot_history, history_sheets = _find_spot_history(sheets, config)
    market = _populate_realized_from_history(market, spot_history)
    meta = pd.DataFrame([{
        "market_data_source": "excel_workbook",
        "market_workbook_path": str(path),
        "workbook_loaded_status": "loaded",
        "sheet_names": ", ".join(sheets.keys()),
        "market_data_sheets": ", ".join(market_sheets),
        "spot_history_sheets": ", ".join(history_sheets),
    }])
    return market, spot_history, meta


def _forward_sheet_meta(sheet_name: str) -> tuple[str, str, str, str]:
    lowered = normalize_header(sheet_name)
    if "eur" in lowered:
        currency_pair = "EUR_TND"
        currency = "EUR"
    elif "usd" in lowered:
        currency_pair = "USD_TND"
        currency = "USD"
    else:
        raise ValueError(f"Unable to infer currency pair from forward sheet name: {sheet_name}")
    if "ask" in lowered:
        side = "ASK"
        direction = "outflow"
    elif "bid" in lowered:
        side = "BID"
        direction = "inflow"
    else:
        raise ValueError(f"Unable to infer ASK/BID side from forward sheet name: {sheet_name}")
    return currency_pair, currency, side, direction


def _detect_forward_header_row(raw: pd.DataFrame, sheet_name: str) -> int:
    for idx in range(min(15, len(raw))):
        value = raw.iloc[idx, 1] if raw.shape[1] > 1 else None
        if isinstance(value, str) and normalize_header(value).startswith("transaction_type"):
            return idx
    raise ValueError(f"Could not find forward sheet header row in {sheet_name}")


def _header_lookup(header: list, patterns: list[str]) -> int | None:
    for idx, value in enumerate(header):
        text = normalize_header(value)
        if any(pattern in text for pattern in patterns):
            return idx
    return None


def load_forward_backtest_long(
    workbook_path: Path,
    tolerance_relative: float = 1e-3,
    run_id: str | None = None,
) -> pd.DataFrame:
    """
    Load the four forward sheets and pivot to long format.

    Returns one row per (transaction_date, currency, side, tenor_months)
    tuple, with both the forward rate locked at hedge_transaction_date
    and the realized spot at transaction_date.
    """

    workbook_path = Path(workbook_path)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Forward workbook not found: {workbook_path}")

    required_sheets = ["EUR ASK Forward", "EUR BID Forward", "USD ASK Forward", "USD BID Forward"]
    tenor_labels = [("1M", 1), ("2M", 2), ("3M", 3), ("6M", 6), ("9M", 9), ("1Y", 12)]
    all_rows: list[pd.DataFrame] = []

    for sheet_name in required_sheets:
        try:
            raw = pd.read_excel(workbook_path, sheet_name=sheet_name, header=None)
        except Exception as exc:
            raise ValueError(f"Failed to read forward sheet {sheet_name}: {exc}") from exc
        if raw.empty:
            raise ValueError(f"Forward sheet {sheet_name} is empty")

        header_row = _detect_forward_header_row(raw, sheet_name)
        header = [normalize_header(v) if not pd.isna(v) else f"unnamed_{i}" for i, v in enumerate(raw.iloc[header_row].tolist())]
        data = raw.iloc[header_row + 1 :].copy()
        data.columns = header
        data = data.dropna(how="all")
        if data.empty:
            raise ValueError(f"Forward sheet {sheet_name} has no data rows after the header")

        transaction_type_col = _header_lookup(header, ["transaction_type"])
        transaction_date_col = _header_lookup(header, ["transaction_date"])
        currency_col = _header_lookup(header, ["currency"])
        historical_spot_col = _header_lookup(header, ["historical_spot", "historical_spot_rate", "spot_history"])
        if transaction_type_col is None or transaction_date_col is None or currency_col is None:
            raise ValueError(f"Forward sheet {sheet_name} is missing one of the required scalar columns")
        if historical_spot_col is None:
            historical_spot_col = 4

        currency_pair, currency_from_sheet, side, direction = _forward_sheet_meta(sheet_name)

        base = pd.DataFrame({
            "transaction_type": data.iloc[:, transaction_type_col],
            "transaction_date": pd.to_datetime(data.iloc[:, transaction_date_col], errors="coerce"),
            "currency": data.iloc[:, currency_col].fillna(currency_from_sheet).astype(str).str.upper(),
            "historical_spot": pd.to_numeric(data.iloc[:, historical_spot_col], errors="coerce"),
        })

        for tenor_label, tenor_months in tenor_labels:
            block_start = None
            for idx, value in enumerate(header):
                text = normalize_header(value)
                if tenor_label.lower() in text and "hedge" in text:
                    block_start = idx
                    break
            if block_start is None or block_start + 5 >= len(header):
                raise ValueError(f"Could not locate {tenor_label} block in forward sheet {sheet_name}")

            block = pd.DataFrame({
                "hedge_transaction_date": pd.to_datetime(data.iloc[:, block_start], errors="coerce"),
                "hedge_n_days": pd.to_numeric(data.iloc[:, block_start + 1], errors="coerce"),
                "hedge_spot_rate": pd.to_numeric(data.iloc[:, block_start + 2], errors="coerce"),
                "domestic_yield": pd.to_numeric(data.iloc[:, block_start + 3], errors="coerce"),
                "foreign_yield": pd.to_numeric(data.iloc[:, block_start + 4], errors="coerce"),
                "forward_rate": pd.to_numeric(data.iloc[:, block_start + 5], errors="coerce"),
            })
            block["currency_pair"] = currency_pair
            block["currency"] = base["currency"]
            block["side"] = side
            block["direction"] = direction
            block["tenor_months"] = tenor_months
            block["transaction_type"] = base["transaction_type"]
            block["transaction_date"] = base["transaction_date"]
            block["historical_spot"] = base["historical_spot"]
            block["realized_spot"] = base["historical_spot"]
            block["realized_forward_advantage"] = np.where(
                block["direction"].eq("outflow"),
                block["realized_spot"] - block["forward_rate"],
                block["forward_rate"] - block["realized_spot"],
            )
            f_recomputed = block["hedge_spot_rate"] * (1.0 + block["domestic_yield"] * block["hedge_n_days"] / 360.0) / (1.0 + block["foreign_yield"] * block["hedge_n_days"] / 360.0)
            block["F_recomputed"] = f_recomputed
            block["cip_recalculation_error"] = (f_recomputed - block["forward_rate"]).abs() / np.maximum(block["forward_rate"].abs(), 1e-12)
            block["cip_recalculation_status"] = np.where(block["cip_recalculation_error"] < tolerance_relative, "ok", "above_tolerance")
            block["source_sheet"] = sheet_name
            if run_id is not None:
                block["run_id"] = run_id
            block = block.dropna(subset=["hedge_transaction_date", "transaction_date", "forward_rate", "realized_spot"])
            all_rows.append(block)

    if not all_rows:
        raise ValueError(f"No forward rows were parsed from workbook {workbook_path}")

    result = pd.concat(all_rows, ignore_index=True)
    if len(result) < 50_000:
        raise ValueError(f"Forward workbook parsing returned too few rows: {len(result)}")
    ordered = [
        "run_id", "source_sheet", "currency_pair", "currency", "side", "direction",
        "tenor_months", "transaction_type", "transaction_date", "hedge_transaction_date",
        "hedge_n_days", "hedge_spot_rate", "domestic_yield", "foreign_yield",
        "forward_rate", "realized_spot", "realized_forward_advantage", "F_recomputed",
        "cip_recalculation_error", "cip_recalculation_status",
    ]
    ordered = [col for col in ordered if col in result.columns]
    result = result[ordered].sort_values(["currency_pair", "side", "tenor_months", "hedge_transaction_date"]).reset_index(drop=True)
    return result


def _populate_realized_from_history(market: pd.DataFrame, spot_history: pd.DataFrame, max_walk_days: int = 7) -> pd.DataFrame:
    if market.empty or spot_history.empty:
        out = market.copy()
        out["realized_lookup_status"] = "missing"
        return out
    out = market.copy()
    out["valuation_date"] = pd.to_datetime(out["valuation_date"], errors="coerce")
    hist = spot_history.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist[hist["date"].notna()].sort_values(["currency_pair", "date"])
    if "realized_future_date" not in out.columns:
        out["realized_future_date"] = pd.NaT
    if "realized_future_spot_bid" not in out.columns:
        out["realized_future_spot_bid"] = np.nan
    if "realized_future_spot_ask" not in out.columns:
        out["realized_future_spot_ask"] = np.nan
    # Force stable dtypes so row-wise assignment of datetimes and floats cannot upcast-fail.
    out["realized_future_date"] = pd.to_datetime(out["realized_future_date"], errors="coerce")
    out["realized_future_spot_bid"] = pd.to_numeric(out["realized_future_spot_bid"], errors="coerce")
    out["realized_future_spot_ask"] = pd.to_numeric(out["realized_future_spot_ask"], errors="coerce")
    status_list = []
    for idx, row in out.iterrows():
        pair = row.get("currency_pair")
        val_date = pd.to_datetime(row.get("valuation_date"), errors="coerce")
        tenor_days = pd.to_numeric(row.get("tenor_days"), errors="coerce")
        if pd.isna(pair) or pd.isna(val_date) or pd.isna(tenor_days):
            status_list.append("missing")
            out.loc[idx, ["realized_future_date", "realized_future_spot_bid", "realized_future_spot_ask"]] = [pd.NaT, np.nan, np.nan]
            continue
        target = val_date + pd.Timedelta(days=int(tenor_days))
        pair_hist = hist[hist["currency_pair"] == pair]
        if pair_hist.empty:
            status_list.append("missing")
            out.loc[idx, ["realized_future_date", "realized_future_spot_bid", "realized_future_spot_ask"]] = [pd.NaT, np.nan, np.nan]
            continue
        candidates = pair_hist[(pair_hist["date"] >= target) & (pair_hist["date"] <= target + pd.Timedelta(days=max_walk_days))]
        if candidates.empty:
            status_list.append("missing")
            out.loc[idx, ["realized_future_date", "realized_future_spot_bid", "realized_future_spot_ask"]] = [pd.NaT, np.nan, np.nan]
            continue
        pick = candidates.iloc[0]
        realized_date = pd.to_datetime(pick["date"])
        walk_days = int((realized_date - target).days)
        if walk_days == 0:
            status = "exact"
        else:
            status = f"walked_forward_{walk_days}"
        out.loc[idx, "realized_future_date"] = realized_date
        out.loc[idx, "realized_future_spot_bid"] = pd.to_numeric(pick["spot_bid"], errors="coerce")
        out.loc[idx, "realized_future_spot_ask"] = pd.to_numeric(pick["spot_ask"], errors="coerce")
        status_list.append(status)
    out["realized_lookup_status"] = status_list
    out["valuation_date"] = out["valuation_date"].dt.strftime("%Y-%m-%d")
    return out
