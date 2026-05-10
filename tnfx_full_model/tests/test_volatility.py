import pandas as pd
import pytest

from tnfx_full_model.volatility import estimate_rolling_volatility


def test_rolling_volatility_tenor_scaling():
    hist = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=10), "currency_pair": "EUR_TND",
                         "spot_mid": [3.0 + i * 0.01 for i in range(10)]})
    market = pd.DataFrame({"currency_pair": ["EUR_TND", "EUR_TND"], "tenor_months": [6, 12], "tenor_days": [180, 360]})
    vol = estimate_rolling_volatility(hist, market, 5)
    assert (vol["sigma_E"] > 0).all()
    assert vol.loc[vol["tenor_days"] == 360, "sigma_E"].iloc[0] > vol.loc[vol["tenor_days"] == 180, "sigma_E"].iloc[0]


def test_rolling_volatility_requires_minimum_rows():
    hist = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=252), "currency_pair": "EUR_TND", "spot_mid": [3.0] * 252})
    market = pd.DataFrame({"currency_pair": ["EUR_TND"], "tenor_months": [6], "tenor_days": [180]})
    with pytest.raises(ValueError, match="Insufficient spot history for 252-day rolling volatility"):
        estimate_rolling_volatility(hist, market, 252)


def test_rolling_volatility_coerces_object_spot_mid():
    hist = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=260),
        "currency_pair": "EUR_TND",
        "spot_mid": [f"{3.0 + i * 0.001:.6f}" for i in range(260)],
    })
    market = pd.DataFrame({"currency_pair": ["EUR_TND"], "tenor_months": [6], "tenor_days": [180]})
    vol = estimate_rolling_volatility(hist, market, 252)
    assert not vol.empty
    assert float(vol["sigma_E"].iloc[0]) > 0
