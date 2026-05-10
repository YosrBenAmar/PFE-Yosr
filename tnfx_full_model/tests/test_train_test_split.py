import pandas as pd

from tnfx_full_model.train_test_split import (
    build_full_walk_forward_splits,
    build_regime_train_test_splits,
    validate_chronological_splits,
    validate_no_train_test_overlap,
)


def _cfg():
    return {
        "backtest_split": {
            "mode": "walk_forward",
            "walk_forward_initial_train_start": "2008-01-01",
            "walk_forward_initial_train_end": "2013-12-31",
            "walk_forward_test_frequency": "yearly",
            "expanding_window": True,
        },
        "regime_backtest": {
            "enabled": True,
            "regimes": {
                "r1": {
                    "label": "r1",
                    "regime_start": "2014-01-01",
                    "regime_end": "2018-12-31",
                    "train_start": "2014-01-01",
                    "train_end": "2016-12-31",
                    "test_start": "2017-01-01",
                    "test_end": "2018-12-31",
                }
            },
        },
    }


def test_walk_forward_insufficient_initial_window_status():
    market = pd.DataFrame({"valuation_date": pd.to_datetime(["2020-01-01", "2021-01-01"]), "currency_pair": ["EUR_TND", "EUR_TND"]})
    splits = build_full_walk_forward_splits(market, _cfg())
    assert splits["split_status"].iloc[0] in {"insufficient_initial_training_window", "insufficient_market_rows"}


def test_walk_forward_expanding_window():
    dates = pd.date_range("2008-01-01", "2025-12-31", freq="180D")
    market = pd.DataFrame({"valuation_date": dates, "currency_pair": ["EUR_TND"] * len(dates)})
    splits = build_full_walk_forward_splits(market, _cfg())
    assert len(splits) >= 2
    assert str(pd.to_datetime(splits.iloc[0]["train_start"]).date()) == "2008-01-01"
    assert str(pd.to_datetime(splits.iloc[1]["train_start"]).date()) == "2008-01-01"
    assert pd.to_datetime(splits.iloc[1]["train_end"]) > pd.to_datetime(splits.iloc[0]["train_end"])


def test_regime_splits_chronological_and_no_overlap():
    dates = pd.date_range("2014-01-01", "2018-12-31", freq="90D")
    market = pd.DataFrame({"valuation_date": dates, "currency_pair": ["EUR_TND"] * len(dates)})
    splits = build_regime_train_test_splits(market, _cfg())
    ok1, _ = validate_chronological_splits(splits)
    ok2, _ = validate_no_train_test_overlap(splits)
    assert ok1
    assert ok2
