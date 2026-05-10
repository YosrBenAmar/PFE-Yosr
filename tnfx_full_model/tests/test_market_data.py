from pathlib import Path

import pandas as pd

from tnfx_full_model.excel_market_loader import load_excel_market_workbook
from tnfx_full_model.market_data import load_market_data


def test_market_data_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"valuation_date": ["2026-01-01"]}).to_csv(p, index=False)
    try:
        load_market_data(p, "2026-01-01")
    except ValueError as exc:
        assert "missing required columns" in str(exc)
    else:
        raise AssertionError("expected missing column error")


def test_excel_loader_normalizes_workbook(tmp_path, config):
    p = tmp_path / "market.xlsx"
    market = pd.DataFrame({
        "Valuation Date": ["2026-01-01"],
        "Currency Pair": ["EUR/TND"],
        "Tenor Months": [6],
        "Tenor Days": [180],
        "Spot Bid": [3.37],
        "Spot Ask": [3.39],
        "Spot Mid": [3.38],
        "TND Rate Bid": [0.07],
        "TND Rate Ask": [0.08],
        "FCY Rate Bid": [0.02],
        "FCY Rate Ask": [0.03],
    })
    hist = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=253),
        "Currency Pair": ["EUR/TND"] * 253,
        "Spot Bid": [3.0] * 253,
        "Spot Ask": [3.02] * 253,
        "Spot Mid": [3.01] * 253,
    })
    with pd.ExcelWriter(p) as writer:
        market.to_excel(writer, sheet_name="Market Data", index=False)
        hist.to_excel(writer, sheet_name="Spot History", index=False)
    mkt, history, meta = load_excel_market_workbook(p, config.market, "2026-01-01")
    assert not mkt.empty
    assert not history.empty
    assert meta["workbook_loaded_status"].iloc[0] == "loaded"


def test_excel_loader_missing_columns_clear_error(tmp_path, config):
    p = tmp_path / "bad.xlsx"
    with pd.ExcelWriter(p) as writer:
        pd.DataFrame({"foo": [1]}).to_excel(writer, sheet_name="Bad", index=False)
    try:
        load_excel_market_workbook(p, config.market, "2026-01-01")
    except ValueError as exc:
        assert "Workbook does not contain enough required market data" in str(exc)
    else:
        raise AssertionError("expected workbook parser error")


def test_realized_future_spot_lookup_at_2024_valuation_date(tmp_path, config):
    p = tmp_path / "market_2024.xlsx"
    rows = []
    for pair in ["EUR/TND", "USD/TND"]:
        for m, d in [(1, 30), (2, 60), (3, 90), (6, 180), (9, 270), (12, 360)]:
            rows.append({
                "valuation_date": "2024-01-01", "currency_pair": pair, "tenor_months": m, "tenor_days": d,
                "spot_bid": 3.30, "spot_ask": 3.32, "spot_mid": 3.31,
                "tnd_rate_bid": 0.07, "tnd_rate_ask": 0.08, "fcy_rate_bid": 0.02, "fcy_rate_ask": 0.03,
            })
    market = pd.DataFrame(rows)
    hist_rows = []
    for pair in ["EUR/TND", "USD/TND"]:
        for dt in pd.date_range("2024-01-01", "2025-01-10", freq="D"):
            hist_rows.append({"date": dt, "currency_pair": pair, "spot_bid": 3.2, "spot_ask": 3.22, "spot_mid": 3.21})
    hist = pd.DataFrame(hist_rows)
    with pd.ExcelWriter(p) as writer:
        market.to_excel(writer, sheet_name="Market Data", index=False)
        hist.to_excel(writer, sheet_name="Spot History", index=False)
    mkt, _, _ = load_excel_market_workbook(p, config.market, "2024-01-01")
    assert len(mkt) == 12
    assert mkt["realized_future_spot_bid"].notna().all()
    assert (mkt["realized_lookup_status"] != "missing").all()


def test_walk_forward_lookup_caps_at_seven_days(tmp_path, config):
    p = tmp_path / "market_gap.xlsx"
    market = pd.DataFrame([{
        "valuation_date": "2024-01-01", "currency_pair": "EUR/TND", "tenor_months": 1, "tenor_days": 30,
        "spot_bid": 3.30, "spot_ask": 3.32, "spot_mid": 3.31,
        "tnd_rate_bid": 0.07, "tnd_rate_ask": 0.08, "fcy_rate_bid": 0.02, "fcy_rate_ask": 0.03,
    }])
    # Target date is 2024-01-31; first available is 2024-02-10 (+10 days) -> must be missing.
    hist = pd.DataFrame([
        {"date": "2024-01-01", "currency_pair": "EUR/TND", "spot_bid": 3.2, "spot_ask": 3.22, "spot_mid": 3.21},
        {"date": "2024-02-10", "currency_pair": "EUR/TND", "spot_bid": 3.25, "spot_ask": 3.27, "spot_mid": 3.26},
    ])
    with pd.ExcelWriter(p) as writer:
        market.to_excel(writer, sheet_name="Market Data", index=False)
        hist.to_excel(writer, sheet_name="Spot History", index=False)
    mkt, _, _ = load_excel_market_workbook(p, config.market, "2024-01-01")
    assert mkt["realized_lookup_status"].iloc[0] == "missing"
    assert pd.isna(mkt["realized_future_spot_bid"].iloc[0])
