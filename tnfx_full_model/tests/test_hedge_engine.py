import pandas as pd

from tnfx_full_model.forward_pricing import price_market_snapshot
from tnfx_full_model.hedge_engine import compute_hedge_decisions


def test_hedge_rules():
    handoff = pd.DataFrame([dict(profile_id=1, family="importer", currency="EUR", currency_pair="EUR_TND",
                                 timing_cv_scenario="baseline", tenor_months=6, E_t=-0.1,
                                 direction="outflow", delta_net_k=-0.1, **{"lambda": 0.6}, sigma_E=0.1, rho=0.0, sigma_Q=0.1)])
    market = pd.DataFrame([dict(valuation_date="2026-01-01", currency_pair="EUR_TND", tenor_months=6, tenor_days=180,
                                spot_bid=3.37, spot_ask=3.39, spot_mid=3.38, tnd_rate_bid=0.07, tnd_rate_ask=0.08,
                                fcy_rate_bid=0.02, fcy_rate_ask=0.03, forward_bid=float("nan"), forward_ask=float("nan"),
                                realized_future_date="2026-07-01", realized_future_spot_bid=3.4, realized_future_spot_ask=3.42)])
    snap = price_market_snapshot(market, pd.DataFrame([dict(currency_pair="EUR_TND", tenor_months=6, tenor_days=180, sigma_E=0.1)]))
    dec = compute_hedge_decisions(handoff, snap, "2026-01-01", {"no_hedge": 0, "baseline_protection": 0.5, "full_hedge": 1}, {"baseline_protection": 100})
    assert dec.loc[dec["hedge_intensity_scenario"] == "no_hedge", "h_c"].iloc[0] == 0
    assert dec.loc[dec["hedge_intensity_scenario"] == "full_hedge", "h_c"].iloc[0] == 0.6
    assert ((dec["h_c"] >= 0) & (dec["h_c"] <= dec["lambda"])).all()
    assert set(dec["binding_constraint"]).issubset({"lower", "upper", "none", "immaterial"})
