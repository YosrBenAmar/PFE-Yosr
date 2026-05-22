import pandas as pd
from pandas.testing import assert_frame_equal

from tnfx_full_model.rolling_summaries import (
    compute_hedge_decision_recommendations,
    compute_strategy_ranking,
)


def _rows_for_scenario(
    scenario: str,
    hedged_pnl: list[float],
    he_t: list[float],
    *,
    currency_pair: str = "EUR_TND",
    side: str = "ASK",
    direction: str = "outflow",
    tenor_months: int = 6,
    regime_label: str = "flexible_managed_regime",
    stress: str = "cip_base",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "currency_pair": [currency_pair] * len(hedged_pnl),
            "side": [side] * len(hedged_pnl),
            "direction": [direction] * len(hedged_pnl),
            "tenor_months": [tenor_months] * len(hedged_pnl),
            "regime_label": [regime_label] * len(hedged_pnl),
            "forward_stress_scenario": [stress] * len(hedged_pnl),
            "hedge_intensity_scenario": [scenario] * len(hedged_pnl),
            "forward_advantage": hedged_pnl,
            "hedge_spot_rate": [3.3] * len(hedged_pnl),
            "hit_indicator": [1 if x > 0 else 0 for x in hedged_pnl],
            "hedged_pnl_per_unit": hedged_pnl,
            "HE_t": he_t,
        }
    )


def test_strategy_ranking_is_deterministic():
    df = pd.concat(
        [
            _rows_for_scenario("no_hedge", [0.00, 0.00, 0.00], [0.00, 0.00, 0.00]),
            _rows_for_scenario("baseline_protection", [0.02, 0.01, -0.01], [0.20, 0.15, 0.10]),
            _rows_for_scenario("high_protection", [0.01, 0.00, -0.02], [0.10, 0.08, 0.05]),
        ],
        ignore_index=True,
    )
    r1 = compute_strategy_ranking(df)
    r2 = compute_strategy_ranking(df)
    assert_frame_equal(r1, r2)


def test_positive_protection_can_outrank_no_hedge():
    df = pd.concat(
        [
            _rows_for_scenario("no_hedge", [0.000, 0.000, 0.000], [0.00, 0.00, 0.00]),
            _rows_for_scenario("baseline_protection", [0.030, 0.020, 0.010], [0.40, 0.35, 0.30]),
            _rows_for_scenario("high_protection", [0.010, 0.005, -0.005], [0.12, 0.10, 0.08]),
        ],
        ignore_index=True,
    )
    ranked = compute_strategy_ranking(df)
    top = ranked[ranked["rank"] == 1].iloc[0]
    assert top["hedge_intensity_scenario"] == "baseline_protection"


def test_no_hedge_can_win_when_active_hedges_are_poor():
    df = pd.concat(
        [
            _rows_for_scenario("no_hedge", [0.0, 0.0, 0.0], [0.00, 0.00, 0.00]),
            _rows_for_scenario("baseline_protection", [-0.05, -0.03, -0.04], [-0.40, -0.30, -0.35]),
            _rows_for_scenario("high_protection", [-0.06, -0.02, -0.05], [-0.55, -0.25, -0.45]),
        ],
        ignore_index=True,
    )
    ranked = compute_strategy_ranking(df)
    top = ranked[ranked["rank"] == 1].iloc[0]
    assert top["hedge_intensity_scenario"] == "no_hedge"


def test_recommendations_no_duplicate_keys():
    stage2 = pd.DataFrame(
        [
            {
                "profile_id": 1, "family": "importer", "currency": "EUR", "currency_pair": "EUR_TND",
                "tenor_months": 6, "timing_cv_scenario": "baseline", "direction": "outflow",
                "pricing_side": "ask", "hedge_intensity_scenario": "no_hedge",
                "E_t": -0.20, "lambda": 0.80, "h_c": 0.0, "expected_cost": 0.0000, "HE_t": 0.00,
            },
            {
                "profile_id": 1, "family": "importer", "currency": "EUR", "currency_pair": "EUR_TND",
                "tenor_months": 6, "timing_cv_scenario": "baseline", "direction": "outflow",
                "pricing_side": "ask", "hedge_intensity_scenario": "baseline_protection",
                "E_t": -0.20, "lambda": 0.80, "h_c": 0.55, "expected_cost": 0.0030, "HE_t": 0.22,
            },
            {
                "profile_id": 2, "family": "processor", "currency": "USD", "currency_pair": "USD_TND",
                "tenor_months": 3, "timing_cv_scenario": "baseline", "direction": "inflow",
                "pricing_side": "bid", "hedge_intensity_scenario": "no_hedge",
                "E_t": 0.15, "lambda": 0.60, "h_c": 0.0, "expected_cost": 0.0000, "HE_t": 0.00,
            },
            {
                "profile_id": 2, "family": "processor", "currency": "USD", "currency_pair": "USD_TND",
                "tenor_months": 3, "timing_cv_scenario": "baseline", "direction": "inflow",
                "pricing_side": "bid", "hedge_intensity_scenario": "high_protection",
                "E_t": 0.15, "lambda": 0.60, "h_c": 0.50, "expected_cost": 0.0040, "HE_t": 0.30,
            },
        ]
    )
    ranking = pd.DataFrame(
        [
            {
                "currency_pair": "EUR_TND", "side": "ASK", "tenor_months": 6, "regime_label": "flexible_managed_regime",
                "forward_stress_scenario": "cip_base", "hedge_intensity_scenario": "baseline_protection",
                "corporate_hedging_score": 0.8, "rank": 1, "is_recommended": True,
                "mean_HE_t": 0.20, "hit_ratio": 0.65, "mean_hedged_pnl_per_unit": 0.01,
                "recommendation_reason": "top corporate score",
            },
            {
                "currency_pair": "EUR_TND", "side": "ASK", "tenor_months": 6, "regime_label": "flexible_managed_regime",
                "forward_stress_scenario": "cip_base", "hedge_intensity_scenario": "no_hedge",
                "corporate_hedging_score": 0.3, "rank": 2, "is_recommended": False,
                "mean_HE_t": 0.00, "hit_ratio": 0.50, "mean_hedged_pnl_per_unit": 0.00,
                "recommendation_reason": "lower score",
            },
        ]
    )
    recs = compute_hedge_decision_recommendations(stage2, ranking)
    keys = ["profile_id", "currency_pair", "tenor_months", "timing_cv_scenario", "direction"]
    assert not recs.empty
    assert not recs.duplicated(keys).any()


def test_recommended_hedged_amount_formula_is_correct():
    stage2 = pd.DataFrame(
        [
            {
                "profile_id": 1, "family": "importer", "currency": "EUR", "currency_pair": "EUR_TND",
                "tenor_months": 6, "timing_cv_scenario": "baseline", "direction": "outflow",
                "pricing_side": "ask", "hedge_intensity_scenario": "baseline_protection",
                "E_t": -0.40, "lambda": 0.75, "h_c": 0.50, "expected_cost": 0.0060, "HE_t": 0.25,
            }
        ]
    )
    ranking = pd.DataFrame(
        [
            {
                "currency_pair": "EUR_TND", "side": "ASK", "tenor_months": 6, "regime_label": "flexible_managed_regime",
                "forward_stress_scenario": "cip_base", "hedge_intensity_scenario": "baseline_protection",
                "corporate_hedging_score": 0.7, "rank": 1, "is_recommended": True,
                "mean_HE_t": 0.25, "hit_ratio": 0.60, "mean_hedged_pnl_per_unit": 0.01,
                "recommendation_reason": "top corporate score",
            }
        ]
    )
    recs = compute_hedge_decision_recommendations(stage2, ranking)
    assert len(recs) == 1
    row = recs.iloc[0]
    assert abs(row["recommended_hedged_amount"] - row["recommended_hedge_ratio"] * abs(row["E_t"])) <= 1e-12


def test_recommended_hedge_ratio_respects_lambda():
    stage2 = pd.DataFrame(
        [
            {
                "profile_id": 7, "family": "trader", "currency": "USD", "currency_pair": "USD_TND",
                "tenor_months": 9, "timing_cv_scenario": "baseline", "direction": "inflow",
                "pricing_side": "bid", "hedge_intensity_scenario": "high_protection",
                "E_t": 0.25, "lambda": 0.55, "h_c": 0.55, "expected_cost": 0.0080, "HE_t": 0.18,
            }
        ]
    )
    ranking = pd.DataFrame(
        [
            {
                "currency_pair": "USD_TND", "side": "BID", "tenor_months": 9, "regime_label": "flexible_managed_regime",
                "forward_stress_scenario": "cip_base", "hedge_intensity_scenario": "high_protection",
                "corporate_hedging_score": 0.9, "rank": 1, "is_recommended": True,
                "mean_HE_t": 0.18, "hit_ratio": 0.70, "mean_hedged_pnl_per_unit": 0.02,
                "recommendation_reason": "top corporate score",
            }
        ]
    )
    recs = compute_hedge_decision_recommendations(stage2, ranking)
    row = recs.iloc[0]
    assert row["recommended_hedge_ratio"] <= row["lambda"] + 1e-12
    assert row["recommended_hedge_ratio"] >= -1e-12
