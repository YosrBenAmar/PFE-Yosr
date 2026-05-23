from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from external_validation import (
    DEFAULT_THRESHOLDS,
    _carry_proxy_from_summary,
    _classify_clients,
    _load_colombus,
    _make_realized_spot_lookup,
    _resolve_columns,
    _test2_hedge_ratio_comparison,
    _test3_counterfactual_pnl,
)


def _write_colombus_workbook(path: Path, tx: pd.DataFrame, ib: pd.DataFrame) -> Path:
    with pd.ExcelWriter(path) as writer:
        tx.to_excel(writer, sheet_name="Transactions", index=False, startrow=1)
        ib.to_excel(writer, sheet_name="IB", index=False, startrow=1)
    return path


def _base_tx_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Deal reference": [1, 2, 3, 4],
            "Client": ["A", "A", "B", "C"],
            "Transaction date": pd.to_datetime(["2024-01-05", "2024-01-10", "2024-01-12", "2024-01-12"]),
            "Value date": pd.to_datetime(["2024-02-05", "2024-01-10", "2024-02-12", "2024-01-12"]),
            "Currency": ["EURTND", "USDTND", "EURTND", "CADTND"],
            "Type": ["Buy", "Sell", "Buy", "Buy"],
            "Instrument": ["Forward", "Spot", "Terme", "Spot"],
            "Amount": [100.0, 100.0, 100.0, 100.0],
            "Amt Currency": [np.nan, np.nan, np.nan, np.nan],
            "Rate": [3.3, 3.2, 3.4, 2.9],
            "Hard Currency": [np.nan, np.nan, np.nan, np.nan],
            "Amount in TND": [330.0, 320.0, 340.0, 290.0],
            "Bank": ["x", "x", "x", "x"],
            "DomicilingBank": [np.nan, np.nan, np.nan, np.nan],
            "commission fees": [0.0, 0.0, 0.0, 0.0],
            "Revenue": [0, 0, 0, 0],
            "Mid market": [3.25, 3.15, 3.30, 2.85],
            "% spread loss": [0.0, 0.0, 0.0, 0.0],
            "Spot": [3.2, np.nan, 3.25, np.nan],
            "Notes": [np.nan, np.nan, np.nan, np.nan],
        }
    )


def _base_ib_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            " Date ": pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-10", "2024-01-12", "2024-02-05", "2024-02-12"]),
            "USDTND": [3.10, 3.11, 3.12, 3.13, 3.15, 3.17],
            "EURTND": [3.30, 3.31, 3.32, 3.33, 3.35, 3.36],
        }
    )


def test_loader_filters_to_EURTND_USDTND_only(tmp_path: Path):
    tx = _base_tx_rows()
    ib = _base_ib_rows()
    wb = _write_colombus_workbook(tmp_path / "colombus.xlsx", tx, ib)
    loaded_tx, _ = _load_colombus(wb)
    assert set(loaded_tx["Currency"].unique()) <= {"EURTND", "USDTND"}


def test_loader_drops_options(tmp_path: Path):
    tx = _base_tx_rows()
    tx.loc[len(tx)] = tx.iloc[0]
    tx.loc[len(tx) - 1, "Instrument"] = "option"
    ib = _base_ib_rows()
    wb = _write_colombus_workbook(tmp_path / "colombus.xlsx", tx, ib)
    loaded_tx, _ = _load_colombus(wb)
    assert not loaded_tx["Instrument"].astype(str).str.lower().eq("option").any()


def test_loader_treats_terme_and_forward_as_same_class(tmp_path: Path):
    tx = _base_tx_rows()
    tx = tx.iloc[[0, 2]].copy()
    ib = _base_ib_rows()
    wb = _write_colombus_workbook(tmp_path / "colombus.xlsx", tx, ib)
    loaded_tx, _ = _load_colombus(wb)
    assert set(loaded_tx["instrument_class"].unique()) == {"hedge"}


def test_classify_clients_pure_importer():
    tx = pd.DataFrame(
        {
            "Client": ["imp"] * 10,
            "Type": ["Buy"] * 10,
            "Currency": ["EURTND"] * 10,
            "Amount_TND": [1000.0] * 10,
            "instrument_class": ["spot"] * 10,
            "tenor_days": [30] * 10,
        }
    )
    uni = _classify_clients(tx, min_deals=10, thresholds=DEFAULT_THRESHOLDS)
    assert uni.loc[0, "family"] == "importer"


def test_classify_clients_pure_exporter():
    tx = pd.DataFrame(
        {
            "Client": ["exp"] * 10,
            "Type": ["Sell"] * 10,
            "Currency": ["USDTND"] * 10,
            "Amount_TND": [1000.0] * 10,
            "instrument_class": ["spot"] * 10,
            "tenor_days": [30] * 10,
        }
    )
    uni = _classify_clients(tx, min_deals=10, thresholds=DEFAULT_THRESHOLDS)
    assert uni.loc[0, "family"] == "exporter"


def test_classify_clients_processor_vs_trader():
    tx = pd.DataFrame(
        {
            "Client": ["proc"] * 10 + ["trad"] * 10,
            "Type": (["Buy"] * 5 + ["Sell"] * 5) * 2,
            "Currency": ["EURTND"] * 20,
            "Amount_TND": [1_000_000.0] * 10 + [100_000.0] * 10,
            "instrument_class": ["hedge"] * 20,
            "tenor_days": [90] * 20,
        }
    )
    uni = _classify_clients(tx, min_deals=10, thresholds=DEFAULT_THRESHOLDS).set_index("client")
    assert uni.loc["proc", "family"] == "processor"
    assert uni.loc["trad", "family"] == "trader"


def test_classify_clients_drops_low_activity():
    tx = pd.DataFrame(
        {
            "Client": ["few"] * 5,
            "Type": ["Buy"] * 5,
            "Currency": ["EURTND"] * 5,
            "Amount_TND": [1000.0] * 5,
            "instrument_class": ["spot"] * 5,
            "tenor_days": [30] * 5,
        }
    )
    uni = _classify_clients(tx, min_deals=10, thresholds=DEFAULT_THRESHOLDS)
    assert bool(uni.loc[0, "included_in_tests"]) is False


def test_realized_spot_returns_prior_business_day_for_missing_date():
    ib = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"]),
            "EURTND": [3.20, 3.30, 3.31],
            "USDTND": [3.10, 3.11, 3.12],
        }
    )
    lookup = _make_realized_spot_lookup(ib)
    sunday = pd.Timestamp("2024-01-07")
    assert lookup(sunday, "EUR_TND") == pytest.approx(3.20)
    with pytest.raises(ValueError, match="before first IB observation"):
        lookup(pd.Timestamp("2024-01-01"), "EUR_TND")


def test_test2_gap_computation_signs():
    tx = pd.DataFrame(
        {
            "Client": ["C1", "C1"],
            "Transaction date": pd.to_datetime(["2024-01-15", "2024-01-15"]),
            "currency_pair": ["EUR_TND", "EUR_TND"],
            "Amount_TND": [300.0, 700.0],
            "instrument_class": ["hedge", "spot"],
            "tenor_days": [90, 0],
            "side": ["ASK", "ASK"],
            "Type": ["Buy", "Buy"],
        }
    )
    uni = pd.DataFrame(
        {
            "client": ["C1"],
            "family": ["importer"],
            "included_in_tests": [True],
            "n_deals": [12],
            "total_tnd": [1000.0],
            "buy_share": [1.0],
            "sell_share": [0.0],
            "eur_share": [1.0],
            "usd_share": [0.0],
            "observed_hedge_ratio": [0.3],
            "mean_tenor_days": [90.0],
            "n_hedge_deals": [1],
        }
    )
    sr = pd.DataFrame(
        {
            "rank": [1],
            "currency_pair": ["EUR_TND"],
            "tenor_months": [3],
            "side": ["ASK"],
            "forward_stress_scenario": ["cip_base"],
            "hedge_intensity_scenario": ["baseline_protection"],
        }
    )
    hdr = pd.DataFrame(
        {
            "family": ["importer", "importer"],
            "currency_pair": ["EUR_TND", "EUR_TND"],
            "tenor_months": [3, 3],
            "timing_cv_scenario": ["baseline", "baseline"],
            "selected_hedge_intensity_scenario": ["baseline_protection", "baseline_protection"],
            "recommended_hedge_ratio": [0.5, 0.5],
        }
    )
    sr_cols = _resolve_columns(
        sr,
        {
            "rank": ["rank"],
            "currency_pair": ["currency_pair"],
            "tenor_months": ["tenor_months"],
            "side": ["side"],
            "forward_stress_scenario": ["forward_stress_scenario"],
            "hedge_intensity_scenario": ["hedge_intensity_scenario"],
        },
        "strategy",
    )
    hdr_cols = _resolve_columns(
        hdr,
        {
            "family": ["family"],
            "currency_pair": ["currency_pair"],
            "tenor_months": ["tenor_months"],
            "timing_cv_scenario": ["timing_cv_scenario"],
            "selected_hedge_intensity_scenario": ["selected_hedge_intensity_scenario"],
            "recommended_hedge_ratio": ["recommended_hedge_ratio"],
        },
        "recommendations",
    )
    out = _test2_hedge_ratio_comparison(tx, uni, sr, sr_cols, hdr, hdr_cols)
    d = out["detail"].iloc[0]
    assert d["gap"] == pytest.approx(0.2)
    assert bool(d["within_pm20"]) is True


def test_test3_buy_improvement_sign():
    tx = pd.DataFrame(
        {
            "Client": ["C1"],
            "Transaction date": pd.to_datetime(["2024-01-31"]),
            "Value date": pd.to_datetime(["2024-01-31"]),
            "Currency": ["EURTND"],
            "currency_pair": ["EUR_TND"],
            "Type": ["Buy"],
            "side": ["ASK"],
            "direction": ["outflow"],
            "Instrument": ["Spot"],
            "instrument_class": ["spot"],
            "Amount": [100.0],
            "Rate": [3.40],
            "Mid market": [3.35],
            "Amount_TND": [340.0],
            "tenor_days": [0],
        }
    )
    uni = pd.DataFrame(
        {
            "client": ["C1"],
            "family": ["importer"],
            "included_in_tests": [True],
            "n_deals": [12],
            "total_tnd": [340.0],
            "buy_share": [1.0],
            "sell_share": [0.0],
            "eur_share": [1.0],
            "usd_share": [0.0],
            "observed_hedge_ratio": [0.0],
            "mean_tenor_days": [0.0],
            "n_hedge_deals": [0],
        }
    )
    ib = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-31"]),
            "EURTND": [3.30, 3.35],
            "USDTND": [3.10, 3.12],
        }
    )
    rs = pd.DataFrame(
        {
            "currency_pair": ["EUR_TND"],
            "side": ["ASK"],
            "tenor_months": [1],
            "hedge_intensity_scenario": ["baseline_protection"],
            "forward_stress_scenario": ["cip_base"],
            "mean_forward_advantage_bps": [0.0],
        }
    )
    sr = pd.DataFrame(
        {
            "rank": [1],
            "currency_pair": ["EUR_TND"],
            "tenor_months": [1],
            "side": ["ASK"],
            "forward_stress_scenario": ["cip_base"],
            "hedge_intensity_scenario": ["baseline_protection"],
        }
    )
    hdr = pd.DataFrame(
        {
            "family": ["importer"],
            "currency_pair": ["EUR_TND"],
            "tenor_months": [1],
            "timing_cv_scenario": ["baseline"],
            "selected_hedge_intensity_scenario": ["baseline_protection"],
            "recommended_hedge_ratio": [0.0],
        }
    )
    sr_cols = _resolve_columns(
        sr,
        {
            "rank": ["rank"],
            "currency_pair": ["currency_pair"],
            "tenor_months": ["tenor_months"],
            "side": ["side"],
            "forward_stress_scenario": ["forward_stress_scenario"],
            "hedge_intensity_scenario": ["hedge_intensity_scenario"],
        },
        "strategy",
    )
    hdr_cols = _resolve_columns(
        hdr,
        {
            "family": ["family"],
            "currency_pair": ["currency_pair"],
            "tenor_months": ["tenor_months"],
            "timing_cv_scenario": ["timing_cv_scenario"],
            "selected_hedge_intensity_scenario": ["selected_hedge_intensity_scenario"],
            "recommended_hedge_ratio": ["recommended_hedge_ratio"],
        },
        "recommendations",
    )
    rs_cols = _resolve_columns(
        rs,
        {
            "currency_pair": ["currency_pair"],
            "tenor_months": ["tenor_months"],
            "side": ["side"],
            "hedge_intensity_scenario": ["hedge_intensity_scenario"],
            "forward_stress_scenario": ["forward_stress_scenario"],
            "mean_forward_advantage": ["mean_forward_advantage_bps", "mean_forward_advantage"],
        },
        "rolling",
    )
    out = _test3_counterfactual_pnl(tx, uni, ib, rs, sr, sr_cols, hdr, hdr_cols, rs_cols)
    detail = out["detail"].iloc[0]
    assert detail["improvement_mid"] > 0


def test_resolve_columns_finds_case_insensitive_match():
    df = pd.DataFrame(columns=["Rank", "Currency_Pair", "Tenor"])
    out = _resolve_columns(
        df,
        {"rank": ["rank"], "currency_pair": ["currency_pair"], "tenor_months": ["tenor"]},
        "test",
    )
    assert out["rank"] == "Rank"
    assert out["currency_pair"] == "Currency_Pair"
    assert out["tenor_months"] == "Tenor"


def test_resolve_columns_raises_clear_error_on_missing():
    df = pd.DataFrame(columns=["foo", "bar"])
    with pytest.raises(ValueError, match="no column matching"):
        _resolve_columns(df, {"rank": ["rank", "Rank"]}, "test")


def test_carry_proxy_handles_bps_vs_fraction():
    rs_bps = pd.DataFrame(
        {
            "currency_pair": ["EUR_TND"],
            "tenor_months": [3],
            "side": ["ASK"],
            "hedge_intensity_scenario": ["baseline_protection"],
            "forward_stress_scenario": ["cip_base"],
            "mean_forward_advantage_bps": [50.0],
        }
    )
    cols_bps = _resolve_columns(
        rs_bps,
        {
            "currency_pair": ["currency_pair"],
            "tenor_months": ["tenor_months"],
            "side": ["side"],
            "hedge_intensity_scenario": ["hedge_intensity_scenario"],
            "forward_stress_scenario": ["forward_stress_scenario"],
            "mean_forward_advantage": ["mean_forward_advantage_bps", "mean_forward_advantage"],
        },
        "rolling_bps",
    )
    carry_bps = _carry_proxy_from_summary(rs_bps, cols_bps, "EUR_TND", 3, "ASK")
    assert carry_bps == pytest.approx(0.005, abs=1e-12)

    rs_frac = rs_bps.rename(columns={"mean_forward_advantage_bps": "mean_forward_advantage"}).copy()
    rs_frac["mean_forward_advantage"] = 0.005
    cols_frac = _resolve_columns(
        rs_frac,
        {
            "currency_pair": ["currency_pair"],
            "tenor_months": ["tenor_months"],
            "side": ["side"],
            "hedge_intensity_scenario": ["hedge_intensity_scenario"],
            "forward_stress_scenario": ["forward_stress_scenario"],
            "mean_forward_advantage": ["mean_forward_advantage_bps", "mean_forward_advantage"],
        },
        "rolling_frac",
    )
    carry_frac = _carry_proxy_from_summary(rs_frac, cols_frac, "EUR_TND", 3, "ASK")
    assert carry_frac == pytest.approx(0.005, abs=1e-12)
