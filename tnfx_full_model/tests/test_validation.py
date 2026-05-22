import pandas as pd

from tnfx_full_model.schemas import ACCEPTED_PROFILE_COLUMNS, HANDOFF_COLUMNS
from tnfx_full_model.validation import validate_stage1
from tnfx_full_model.validation import validate_stage2


def test_validation_table_and_violation(config):
    tables = {
        "Stage_2_Decisions": pd.DataFrame([dict(h_c=2.0, **{"lambda": 1.0}, hedge_intensity_scenario="x", direction="outflow", pricing_side="ask")]),
        "Market_Data_Snapshot": pd.DataFrame([dict(spot_ask=3.39, tnd_rate_ask=0.08, fcy_rate_bid=0.03, tenor_days=180, F_CIP_ask=0.0)]),
        "Backtest_Results": pd.DataFrame([dict(n_observations=0)]),
    }
    checks = validate_stage2(tables, config.market)
    assert not checks.empty
    assert (checks["result"] == "fail").any()
    assert "no_option" in checks["check_name"].iloc[-1]


def test_stage1_validation_catches_missing_handoff(config, small_run):
    profile = {col: 0 for col in ACCEPTED_PROFILE_COLUMNS}
    profile.update({
        "profile_id": 1, "family": "importer", "alpha_R_EUR": 0.5, "alpha_R_USD": 0.5,
        "alpha_C_EUR": 0.5, "alpha_C_USD": 0.5, "alpha_D_EUR": 0.5, "alpha_D_USD": 0.5,
        "g_EXT": 0,
    })
    tables = {
        "Accepted_Profiles": pd.DataFrame([profile]),
        "Inactive_Profiles": pd.DataFrame(),
        "BM_Exposure_Diagnostics": pd.DataFrame([{
            "profile_id": 1, "family": "importer", "delta_CF_total": -0.06,
            "fx_debt_service_share_EUR": 0.0, "fx_debt_service_share_USD": 0.0,
            "delta_net_EUR": -0.02, "delta_net_USD": 0.0,
        }]),
        "Stage_1_5_Handoff": pd.DataFrame(columns=HANDOFF_COLUMNS),
        "Tenor_Weights": pd.DataFrame([{"profile_id": 1, "timing_cv_scenario": "baseline", "omega_t": 1.0}]),
        "Macro_Anchor_Check": pd.DataFrame([{"prior_mode": "nominal", "pass_fail": "pass"}]),
        "Sobol_Acceptance": pd.DataFrame([{"family": "processor", "acceptance_rate": 1.0}, {"family": "trader", "acceptance_rate": 1.0}]),
    }
    checks = validate_stage1(tables, config.model, small_run)
    row = checks[checks["check_name"] == "every_accepted_profile_has_handoff"].iloc[0]
    assert row["result"] == "fail"


def test_realized_future_spot_check_present_in_validation(config):
    tables = {
        "Stage_2_Decisions": pd.DataFrame([{
            "h_c": 0.0, "lambda": 1.0, "hedge_intensity_scenario": "no_hedge",
            "direction": "outflow", "pricing_side": "ask", "stage2_row_status": "material",
            "HE_t": 0.0, "variance_unhedged": 1.0, "variance_hedged": 1.0, "h_star": 0.0,
            "expected_cost": 0.0, "profile_id": 1, "family": "importer",
        }]),
        "Market_Data_Snapshot": pd.DataFrame([{
            "currency_pair": "EUR_TND", "tenor_months": 6, "tenor_days": 180,
            "spot_ask": 3.39, "tnd_rate_ask": 0.08, "fcy_rate_bid": 0.03, "F_CIP_ask": 3.39,
            "realized_future_spot_bid": 3.4,
        }]),
        "Backtest_Results": pd.DataFrame([{"status": "backtest_completed"}]),
    }
    checks = validate_stage2(tables, config.market)
    assert "realized_future_spot_columns_populated_when_backtest_completed" in set(checks["check_name"])


def test_new_validation_checks_present(config):
    tables = {
        "Stage_2_Decisions": pd.DataFrame([{
            "h_c": 0.5, "lambda": 1.0, "hedge_intensity_scenario": "baseline_protection",
            "direction": "outflow", "pricing_side": "ask", "stage2_row_status": "material",
            "HE_t": 0.1, "variance_unhedged": 1.0, "variance_hedged": 0.9, "h_star": 0.5,
            "expected_cost": 0.01, "profile_id": 1, "family": "importer",
            "S0": 3.3, "sigma_E": 0.2, "rho": 0.1, "sigma_Q": 0.2, "F_executable": 3.4,
            "carry_cost": 0.01, "forward_bias": 0.0, "gamma_R": 1.0, "gamma_R_used": 1.0,
            "gamma_R_source": "global_calibration", "binding_constraint": "none",
            "signed_carry_effect": 0.01, "valuation_date": "2024-01-01",
            "realized_future_date": "2024-07-01", "realized_future_spot_bid": 3.4, "realized_future_spot_ask": 3.5,
        }]),
        "Market_Data_Snapshot": pd.DataFrame([{
            "currency_pair": "EUR_TND", "tenor_months": 6, "tenor_days": 180,
            "spot_ask": 3.39, "tnd_rate_ask": 0.08, "fcy_rate_bid": 0.03, "F_CIP_ask": 3.39,
            "realized_future_spot_bid": 3.4,
        }]),
        "Market_Load_Metadata": pd.DataFrame([{"workbook_loaded_status": "loaded"}]),
        "Spot_History_Counts": pd.DataFrame([{"currency_pair": "EUR_TND", "n_rows": 300}]),
        "Backtest_Results": pd.DataFrame([{"status": "backtest_completed"}]),
        "Forward_Backtest_Long": pd.DataFrame([
            {"currency_pair": "EUR_TND", "side": "ASK", "tenor_months": 1, "cip_recalculation_status": "ok"},
            {"currency_pair": "EUR_TND", "side": "BID", "tenor_months": 2, "cip_recalculation_status": "ok"},
            {"currency_pair": "USD_TND", "side": "ASK", "tenor_months": 3, "cip_recalculation_status": "ok"},
            {"currency_pair": "USD_TND", "side": "BID", "tenor_months": 6, "cip_recalculation_status": "ok"},
            {"currency_pair": "EUR_TND", "side": "ASK", "tenor_months": 9, "cip_recalculation_status": "ok"},
            {"currency_pair": "USD_TND", "side": "BID", "tenor_months": 12, "cip_recalculation_status": "ok"},
        ]),
        "Rolling_Market_Performance": pd.DataFrame([{
            "currency_pair": "EUR_TND", "side": "ASK", "tenor_months": 6, "hedge_intensity_scenario": "baseline_protection",
            "forward_stress_scenario": "cip_base", "market_row_status": "ok", "h_c": 0.5, "HE_t": 0.1,
            "carry_cost_used": 0.01, "forward_advantage": 0.02, "hedge_spot_rate": 3.3,
        }, {
            "currency_pair": "EUR_TND", "side": "ASK", "tenor_months": 6, "hedge_intensity_scenario": "baseline_protection",
            "forward_stress_scenario": "cip_plus_50bps", "market_row_status": "ok", "h_c": 0.6, "HE_t": 0.12,
            "carry_cost_used": 0.02, "forward_advantage": 0.02, "hedge_spot_rate": 3.3,
        }]),
        "Split_OOS_Performance": pd.DataFrame([{
            "split_specific_gamma_used": True, "methodological_status": "true_out_of_sample",
        }]),
        "Gamma_R_Calibration_By_Split": pd.DataFrame([{
            "train_start": "2008-01-01", "train_end": "2013-12-31", "test_start": "2014-01-01",
        }]),
        "Negative_HE_Diagnostics": pd.DataFrame([{"currency_pair": "EUR_TND"}]),
    }
    checks = validate_stage2(tables, config.market)
    names = set(checks["check_name"])
    for expected in {
        "forward_backtest_long_loaded",
        "cip_wedge_changes_h_c",
        "split_specific_gamma_used",
        "first_walk_forward_split_is_2014",
        "negative_he_diagnostics_table_present",
    }:
        assert expected in names


def test_recommendation_validation_checks(config):
    tables = {
        "Stage_2_Decisions": pd.DataFrame([{
            "profile_id": 1, "family": "importer", "currency": "EUR", "currency_pair": "EUR_TND",
            "tenor_months": 6, "timing_cv_scenario": "baseline", "direction": "outflow",
            "h_c": 0.5, "lambda": 0.8, "hedge_intensity_scenario": "baseline_protection",
            "pricing_side": "ask", "stage2_row_status": "material", "HE_t": 0.1,
            "variance_unhedged": 1.0, "variance_hedged": 0.9, "h_star": 0.5,
            "expected_cost": 0.01, "E_t": -0.2,
        }]),
        "Market_Data_Snapshot": pd.DataFrame([{
            "currency_pair": "EUR_TND", "tenor_months": 6, "tenor_days": 180,
            "spot_ask": 3.39, "tnd_rate_ask": 0.08, "fcy_rate_bid": 0.03, "F_CIP_ask": 3.39,
            "realized_future_spot_bid": 3.4,
        }]),
        "Backtest_Results": pd.DataFrame([{"status": "backtest_completed"}]),
        "Hedge_Decision_Recommendations": pd.DataFrame([{
            "profile_id": 1, "family": "importer", "currency": "EUR", "currency_pair": "EUR_TND",
            "tenor_months": 6, "timing_cv_scenario": "baseline", "direction": "outflow",
            "E_t": -0.2, "abs_exposure": 0.2, "lambda": 0.8,
            "selected_hedge_intensity_scenario": "baseline_protection",
            "recommended_hedge_ratio": 0.5, "recommended_hedged_amount": 0.1,
            "expected_cost": 0.01, "mean_HE_t": 0.1, "hit_ratio": 0.6,
            "mean_hedged_pnl_per_unit": 0.01, "recommendation_reason": "ok",
            "recommendation_source": "strategy_ranking",
        }]),
    }
    checks = validate_stage2(tables, config.market)
    names = set(checks["check_name"])
    for expected in {
        "hedge_decision_recommendations_exists",
        "hedge_recommendations_unique_exposure_key",
        "selected_hedge_intensity_scenario_not_null",
        "recommended_hedge_ratio_within_zero_lambda",
        "recommended_hedged_amount_matches_ratio_times_abs_exposure",
        "recommendation_reason_not_empty",
    }:
        assert expected in names


def test_gamma_methodology_validation_checks_present(config):
    tables = {
        "Stage_2_Decisions": pd.DataFrame([{
            "profile_id": 1, "family": "importer", "currency": "EUR", "currency_pair": "EUR_TND",
            "tenor_months": 6, "timing_cv_scenario": "baseline", "direction": "outflow",
            "h_c": 0.4, "lambda": 0.8, "hedge_intensity_scenario": "baseline_protection",
            "pricing_side": "ask", "stage2_row_status": "material", "HE_t": 0.1,
            "variance_unhedged": 1.0, "variance_hedged": 0.9, "h_star": 0.4,
            "expected_cost": 0.01, "E_t": -0.2,
        }]),
        "Market_Data_Snapshot": pd.DataFrame([{
            "currency_pair": "EUR_TND", "tenor_months": 6, "tenor_days": 180,
            "spot_ask": 3.39, "tnd_rate_ask": 0.08, "fcy_rate_bid": 0.03, "F_CIP_ask": 3.39,
            "realized_future_spot_bid": 3.4,
        }]),
        "Backtest_Results": pd.DataFrame([{"status": "backtest_completed"}]),
        "Hedge_Decision_Recommendations": pd.DataFrame([{
            "profile_id": 1, "family": "importer", "currency": "EUR", "currency_pair": "EUR_TND",
            "tenor_months": 6, "timing_cv_scenario": "baseline", "direction": "outflow",
            "E_t": -0.2, "abs_exposure": 0.2, "lambda": 0.8,
            "selected_hedge_intensity_scenario": "baseline_protection",
            "recommended_hedge_ratio": 0.4, "recommended_hedged_amount": 0.08,
            "expected_cost": 0.01, "mean_HE_t": 0.1, "hit_ratio": 0.6,
            "mean_hedged_pnl_per_unit": 0.01, "recommendation_reason": "ok",
            "recommendation_source": "strategy_ranking",
        }]),
        "Gamma_R_Calibration_Global": pd.DataFrame([{
            "hedge_intensity_scenario": "baseline_protection",
            "target_intensity": 0.5,
            "benchmark_family_used": "importer",
            "benchmark_currency_used": "EUR",
            "benchmark_exposure_definition": "median_abs_delta_net_EUR_importer",
            "E_star": 0.05,
            "rho_star": 0.1,
            "sigma_Q_star": 0.2,
            "currency_pair": "EUR_TND",
            "tenor_months": 6,
            "S0_star": 3.4,
            "sigma_E_star": 0.08,
            "carry_cost_star": 0.06,
            "gamma_R": 0.01,
            "calibration_status": "ok",
            "gamma_methodology": "global_population_importer_benchmark",
        }]),
        "Gamma_R_Calibration_By_Split": pd.DataFrame([{
            "split_id": "wf_001",
            "train_end": "2013-12-31",
            "test_start": "2014-01-01",
            "hedge_intensity_scenario": "baseline_protection",
            "calibration_status": "ok",
            "gamma_methodology": "split_specific_market_side",
            "benchmark_exposure_definition": "fixed_unit_exposure",
        }]),
    }
    checks = validate_stage2(tables, config.market)
    names = set(checks["check_name"])
    for expected in {
        "gamma_r_calibration_global_exists",
        "gamma_r_calibration_global_required_columns_present",
        "gamma_r_global_positive_estar_for_active_ok_rows",
        "gamma_r_global_has_methodology_labels",
        "gamma_split_methodology_disclosed_or_status_present",
        "no_false_client_specific_gamma_claims",
    }:
        assert expected in names


def test_output_manifest_validation_checks_present(config):
    tables = {
        "Stage_2_Decisions": pd.DataFrame([{
            "profile_id": 1, "family": "importer", "currency": "EUR", "currency_pair": "EUR_TND",
            "tenor_months": 6, "timing_cv_scenario": "baseline", "direction": "outflow",
            "h_c": 0.4, "lambda": 0.8, "hedge_intensity_scenario": "baseline_protection",
            "pricing_side": "ask", "stage2_row_status": "material", "HE_t": 0.1,
            "variance_unhedged": 1.0, "variance_hedged": 0.9, "h_star": 0.4,
            "expected_cost": 0.01, "E_t": -0.2,
        }]),
        "Market_Data_Snapshot": pd.DataFrame([{
            "currency_pair": "EUR_TND", "tenor_months": 6, "tenor_days": 180,
            "spot_ask": 3.39, "tnd_rate_ask": 0.08, "fcy_rate_bid": 0.03, "F_CIP_ask": 3.39,
            "realized_future_spot_bid": 3.4,
        }]),
        "Backtest_Results": pd.DataFrame([{"status": "backtest_completed"}]),
        "Hedge_Decision_Recommendations": pd.DataFrame([{
            "profile_id": 1, "family": "importer", "currency": "EUR", "currency_pair": "EUR_TND",
            "tenor_months": 6, "timing_cv_scenario": "baseline", "direction": "outflow",
            "E_t": -0.2, "abs_exposure": 0.2, "lambda": 0.8,
            "selected_hedge_intensity_scenario": "baseline_protection",
            "recommended_hedge_ratio": 0.4, "recommended_hedged_amount": 0.08,
            "expected_cost": 0.01, "mean_HE_t": 0.1, "hit_ratio": 0.6,
            "mean_hedged_pnl_per_unit": 0.01, "recommendation_reason": "ok",
            "recommendation_source": "strategy_ranking",
        }]),
        "Output_Manifest": pd.DataFrame([{
            "run_id": "x",
            "table_name": "Stage_2_Decisions",
            "file_name": "Stage_2_Decisions.csv",
            "file_path": "data/outputs/latest/Stage_2_Decisions.csv",
            "n_rows": 1,
            "n_columns": 5,
            "export_status": "exported",
            "export_timestamp": "2026-05-21T10:00:00",
            "error_message": "",
        }]),
        "Gamma_R_Calibration_Global": pd.DataFrame([{
            "hedge_intensity_scenario": "baseline_protection",
            "target_intensity": 0.5,
            "benchmark_family_used": "importer",
            "benchmark_currency_used": "EUR",
            "benchmark_exposure_definition": "median_abs_delta_net_EUR_importer",
            "E_star": 0.05,
            "rho_star": 0.1,
            "sigma_Q_star": 0.2,
            "currency_pair": "EUR_TND",
            "tenor_months": 6,
            "S0_star": 3.4,
            "sigma_E_star": 0.08,
            "carry_cost_star": 0.06,
            "gamma_R": 0.01,
            "calibration_status": "ok",
            "gamma_methodology": "global_population_importer_benchmark",
        }]),
    }
    checks = validate_stage2(tables, config.market)
    names = set(checks["check_name"])
    for expected in {
        "output_manifest_exists",
        "output_manifest_required_columns_present",
        "output_manifest_nonempty_when_csv_export_enabled",
        "output_manifest_failed_exports_explicitly_reported",
    }:
        assert expected in names
