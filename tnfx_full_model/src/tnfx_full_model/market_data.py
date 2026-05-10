from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import MARKET_COLUMNS


def load_market_data(path: str | Path, valuation_date: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing market data file: {path}")
    df = pd.read_csv(path)
    missing = [c for c in MARKET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Market data missing required columns: {missing}")
    df = df.copy()
    df["valuation_date"] = pd.to_datetime(df["valuation_date"]).dt.strftime("%Y-%m-%d")
    df = df[df["valuation_date"] == valuation_date]
    if df.empty:
        raise ValueError(f"No market data rows for valuation_date={valuation_date}")
    if df["spot_mid"].isna().any():
        df["spot_mid"] = (df["spot_bid"] + df["spot_ask"]) / 2.0
    return df


def load_spot_history(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing spot history file: {path}")
    df = pd.read_csv(path)
    required = ["date", "currency_pair", "spot_bid", "spot_ask", "spot_mid"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Spot history missing required columns: {missing}")
    if df.empty:
        raise ValueError("Spot history file has no observations")
    df = df.copy()
    if df["spot_mid"].isna().any():
        df["spot_mid"] = (df["spot_bid"] + df["spot_ask"]) / 2.0
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["currency_pair", "date"])


def load_market_inputs(project_root: Path, market_config: dict, valuation_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = market_config.get("market_data_source", "csv")
    if source == "excel_workbook":
        from .excel_market_loader import load_excel_market_workbook

        path = project_root / market_config["market_workbook_path"]
        return load_excel_market_workbook(path, market_config, valuation_date)
    if source == "csv":
        market_path = project_root / market_config["fallback_market_csv_path"]
        history_path = project_root / market_config["fallback_spot_history_csv_path"]
        if market_path.name == Path(market_config.get("market_data_template_path", "")).name:
            raise ValueError("Templates are not runtime market data inputs. Configure fallback_market_csv_path.")
        if history_path.name == Path(market_config.get("spot_history_template_path", "")).name:
            raise ValueError("Templates are not runtime spot history inputs. Configure fallback_spot_history_csv_path.")
        meta = pd.DataFrame([{
            "market_data_source": "csv",
            "market_workbook_path": "",
            "workbook_loaded_status": "not_applicable",
            "market_data_path": str(market_path),
            "spot_history_path": str(history_path),
        }])
        return load_market_data(market_path, valuation_date), load_spot_history(history_path), meta
    raise ValueError(f"Unsupported market_data_source: {source}")


def join_market(handoff: pd.DataFrame, market_snapshot: pd.DataFrame, valuation_date: str) -> pd.DataFrame:
    market = market_snapshot.copy()
    market["valuation_date"] = valuation_date
    merged = handoff.merge(market, on=["currency_pair", "tenor_months"], how="left", suffixes=("", "_mkt"))
    missing = merged["spot_bid"].isna()
    if missing.any():
        keys = merged.loc[missing, ["currency_pair", "tenor_months"]].drop_duplicates().to_dict("records")
        raise ValueError(f"Missing market rows for handoff keys: {keys[:10]}")
    return merged

def synthetic_history_from_market(market: pd.DataFrame, days: int = 260) -> pd.DataFrame:
    rows = []
    for pair, row in market.drop_duplicates("currency_pair").set_index("currency_pair").iterrows():
        spot = float(row["spot_mid"])
        for i in range(days):
            level = spot * (1.0 - 0.0005 * (days - i))
            rows.append({"date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i), "currency_pair": pair,
                         "spot_bid": level * 0.997, "spot_ask": level * 1.003, "spot_mid": level})
    return pd.DataFrame(rows)
