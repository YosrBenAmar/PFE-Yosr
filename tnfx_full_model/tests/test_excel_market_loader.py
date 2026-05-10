from pathlib import Path

import numpy as np
import pandas as pd

from tnfx_full_model.excel_market_loader import load_forward_backtest_long


def _workbook_path(config) -> Path:
    return config.root / config.market["market_workbook_path"]


def _header_row(sheet: pd.DataFrame) -> int:
    for idx in range(min(15, len(sheet))):
        if str(sheet.iloc[idx, 1]).strip().lower().startswith("transaction type"):
            return idx
    raise AssertionError("header row not found")


def test_load_forward_backtest_long_basic_shape(config):
    df = load_forward_backtest_long(_workbook_path(config), tolerance_relative=1e-3, run_id="unit_test")
    assert len(df) >= 100_000
    required = {
        "run_id", "source_sheet", "currency_pair", "currency", "side", "direction", "tenor_months",
        "transaction_type", "transaction_date", "hedge_transaction_date", "hedge_n_days", "hedge_spot_rate",
        "domestic_yield", "foreign_yield", "forward_rate", "realized_spot", "realized_forward_advantage",
        "F_recomputed", "cip_recalculation_error", "cip_recalculation_status",
    }
    assert required.issubset(df.columns)
    combos = set(map(tuple, df[["currency_pair", "side"]].drop_duplicates().to_records(index=False).tolist()))
    assert combos == {("EUR_TND", "ASK"), ("EUR_TND", "BID"), ("USD_TND", "ASK"), ("USD_TND", "BID")}
    assert set(df["tenor_months"].dropna().astype(int).unique().tolist()) == {1, 2, 3, 6, 9, 12}


def test_load_forward_backtest_long_realized_spot_matches_historical_spot(config):
    workbook = _workbook_path(config)
    df = load_forward_backtest_long(workbook, tolerance_relative=1e-3, run_id="unit_test")
    sample = df.sample(n=5, random_state=7)
    for row in sample.itertuples(index=False):
        raw = pd.read_excel(workbook, sheet_name=row.source_sheet, header=None)
        header_row = _header_row(raw)
        data = raw.iloc[header_row + 1 :].copy()
        dates = pd.to_datetime(data.iloc[:, 2], errors="coerce")
        matches = data[dates == pd.to_datetime(row.transaction_date)]
        assert not matches.empty
        historical_spot = pd.to_numeric(matches.iloc[0, 4], errors="coerce")
        assert np.isclose(float(row.realized_spot), float(historical_spot), atol=1e-10)


def test_load_forward_backtest_long_cip_check(config):
    df = load_forward_backtest_long(_workbook_path(config), tolerance_relative=1e-3, run_id="unit_test")
    share_ok = (df["cip_recalculation_status"] == "ok").mean()
    assert share_ok > 0.95