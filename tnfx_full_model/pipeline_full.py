from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
import shutil
import time

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline_stage1 import run_stage1
from tnfx_full_model.backtest import gamma_calibration_by_split, run_backtest, run_split_backtests
from tnfx_full_model.config_loader import load_project_config
from tnfx_full_model.cohort_attribution import compute_profile_cohort_attribution
from tnfx_full_model.exporters import enrich_reporting_tables, export_tables
from tnfx_full_model.excel_market_loader import load_forward_backtest_long
from tnfx_full_model.forward_pricing import price_market_snapshot
from tnfx_full_model.gamma_calibration import calibrate_gamma_R
from tnfx_full_model.hedge_engine import aggregate_hedge_profile, cohort_analysis, compute_hedge_decisions
from tnfx_full_model.market_data import load_market_inputs
from tnfx_full_model.rolling_market_performance import compute_oos_market_performance, compute_rolling_market_performance, set_regime_breaks
from tnfx_full_model.rolling_summaries import compute_negative_he_diagnostics, compute_regime_performance, compute_rolling_market_summary, compute_strategy_ranking
from tnfx_full_model.sensitivity import sensitivity_summary
from tnfx_full_model.train_test_split import (
    build_full_walk_forward_splits,
    build_regime_train_test_splits,
    validate_chronological_splits,
    validate_no_train_test_overlap,
)
from tnfx_full_model.validation import hard_failures, validate_stage1, validate_stage2
from tnfx_full_model.volatility import estimate_rolling_volatility


def _load_stage1_tables_from_dir(reuse_dir: Path) -> dict[str, pd.DataFrame]:
    required = {
        "Accepted_Profiles",
        "Inactive_Profiles",
        "BM_Exposure_Diagnostics",
        "Timing_CV_Scenarios",
        "Tenor_Weights",
        "Stage_1_5_Handoff",
        "Macro_Anchor_Check",
        "Sign_Threshold_Sensitivity",
        "Sobol_Acceptance",
        "Sobol_Meta",
        "Regime_State_Priors",
    }
    tables: dict[str, pd.DataFrame] = {}
    for csv_path in reuse_dir.glob("*.csv"):
        try:
            tables[csv_path.stem] = pd.read_csv(csv_path)
        except Exception:
            continue
    missing = sorted([name for name in required if name not in tables])
    if missing:
        raise FileNotFoundError(
            f"stage2_only requires Stage 1 tables in '{reuse_dir}'. Missing: {missing}"
        )
    return tables


def run_full(
    mode: str = "stage2_only",
    reuse_stage1_dir: str | None = None,
    reuse_run_id: str | None = None,
    no_clean: bool = False,
    debug_skip_splits: bool = False,
):
    config = load_project_config(ROOT)
    t0 = time.perf_counter()
    def ckpt(label: str) -> None:
        print(f"[CHECKPOINT] {label} | elapsed={time.perf_counter()-t0:.1f}s")
    run_id = reuse_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = config.root / "data" / "outputs" / run_id
    fast_debug = mode == "stage2_only" and debug_skip_splits
    if mode == "stage2_only":
        if not reuse_stage1_dir:
            raise ValueError("stage2_only mode requires --reuse-stage1-dir <path>.")
        source_dir = Path(reuse_stage1_dir)
        if not source_dir.is_absolute():
            source_dir = (config.root / source_dir).resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f"reuse stage1 directory not found: {source_dir}")
        if not no_clean and source_dir.resolve() == run_output_dir.resolve():
            raise ValueError("Refusing to clean: reuse source directory equals output directory. Use --no-clean.")
        if not no_clean and run_output_dir.exists():
            shutil.rmtree(run_output_dir)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        tables = _load_stage1_tables_from_dir(source_dir)
        ckpt(f"stage1 reused from {source_dir}")
    else:
        if not no_clean and run_output_dir.exists():
            shutil.rmtree(run_output_dir)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        tables, _ = run_stage1(run_id=run_id)
        ckpt("stage1 regenerated")
    valuation_date = config.run["valuation_date"]
    market, history, market_meta = load_market_inputs(config.root, config.market, valuation_date)
    tables["Market_Load_Metadata"] = market_meta
    tables["Spot_History_Counts"] = history.groupby("currency_pair").size().reset_index(name="n_rows")
    ckpt(f"market loaded rows={len(market)}, history={len(history)}")
    vol = estimate_rolling_volatility(history, market, config.market["vol_window_days"], config.market.get("vol_method", "rolling"))
    snapshot = price_market_snapshot(market, vol, config.market.get("vol_method", "rolling"), config.market["cip_wedge_bps"])
    tables["Market_Data_Snapshot"] = snapshot
    ckpt(f"snapshot ready rows={len(snapshot)}")
    gamma_R, gamma_detail = calibrate_gamma_R(
        tables["Accepted_Profiles"], tables["BM_Exposure_Diagnostics"], snapshot,
        config.market["hedge_intensity_scenarios"],
    )
    decisions = compute_hedge_decisions(
        tables["Stage_1_5_Handoff"], snapshot, valuation_date,
        config.market["hedge_intensity_scenarios"], gamma_R,
        brown_toft_constant=config.market.get("brown_toft_constant", 0.0),
        min_abs_exposure=config.market.get("stage2_min_abs_exposure", 1.0e-6),
        min_relative_exposure=config.market.get("stage2_min_relative_exposure", 0.001),
    )
    decision_cols = [
        "profile_id", "family", "currency", "currency_pair", "timing_cv_scenario",
        "tenor_months", "hedge_intensity_scenario", "target_intensity", "E_t",
        "direction", "delta_net_k", "materiality_threshold", "stage2_row_status",
        "lambda", "sigma_E", "rho", "sigma_Q", "S0", "F_executable",
        "carry_cost", "forward_bias", "gamma_R", "h_star", "h_c",
        "gamma_R_used", "gamma_R_source",
        "binding_constraint", "expected_cost", "signed_carry_effect",
        "variance_unhedged", "variance_hedged", "HE_t",
    ]
    tables["Stage_2_Decisions"] = decisions[decision_cols + ["pricing_side", "valuation_date", "realized_future_date", "realized_future_spot_bid", "realized_future_spot_ask"]]
    ckpt(f"stage2 decisions rows={len(tables['Stage_2_Decisions'])}")
    if fast_debug:
        tables["Backtest_Results"] = pd.DataFrame([{
            "status": "skipped_for_debug",
            "n_rows_missing_realized_future_spot": 0,
            "n_rows_backtested": 0,
            "missing_realized_spot_warning": False,
            "n_observations": 0,
        }])
        forward_backtest_long = pd.DataFrame([{"run_id": run_id, "status": "skipped_for_debug"}])
        rolling_market = pd.DataFrame([{"run_id": run_id, "market_row_status": "skipped_for_debug"}])
        stress_scenarios = config.market.get("forward_stress_scenarios", {"cip_base": 0, "cip_plus_50bps": 50})
        tables["Forward_Backtest_Long"] = forward_backtest_long
        tables["Rolling_Market_Performance"] = rolling_market
        ckpt("backtest skipped for debug")
        ckpt("rolling skipped for debug")
    else:
        tables["Backtest_Results"] = run_backtest(decisions, config.market["sub_periods"], config.market.get("require_backtest", False))
        ckpt(f"backtest completed rows={len(tables['Backtest_Results'])}")
        set_regime_breaks(config.market.get("regime_breaks"))
        forward_backtest_long = load_forward_backtest_long(
            workbook_path=config.root / config.market["market_workbook_path"],
            tolerance_relative=config.market.get("cip_recalc_tolerance", 1e-3),
            run_id=run_id,
        )
        forward_backtest_long["run_id"] = run_id
        tables["Forward_Backtest_Long"] = forward_backtest_long
        stress_scenarios = config.market.get("forward_stress_scenarios", {"cip_base": 0, "cip_plus_50bps": 50})
        rolling_market = compute_rolling_market_performance(
            forward_backtest_long=forward_backtest_long,
            spot_history_long=history,
            accepted_profiles=tables["Accepted_Profiles"],
            hedge_scenarios=config.market["hedge_intensity_scenarios"],
            gamma_R_global=gamma_R,
            stress_scenarios=stress_scenarios,
            vol_window_days=config.market["vol_window_days"],
            run_id=run_id,
        )
        tables["Rolling_Market_Performance"] = rolling_market
        ckpt(f"rolling completed rows={len(rolling_market)}")
    split_source = history.rename(columns={"date": "valuation_date"})[["valuation_date", "currency_pair"]].copy()
    if fast_debug:
        split_cols = ["split_id", "split_type", "regime_name", "regime_label", "train_start", "train_end", "test_start", "test_end", "status"]
        walk_splits = pd.DataFrame([{
            "split_id": "",
            "split_type": "full_walk_forward",
            "regime_name": "",
            "regime_label": "",
            "train_start": pd.NaT,
            "train_end": pd.NaT,
            "test_start": pd.NaT,
            "test_end": pd.NaT,
            "status": "skipped_for_debug",
        }], columns=split_cols)
        regime_splits = pd.DataFrame([{
            "split_id": "",
            "split_type": "regime_train_test",
            "regime_name": "",
            "regime_label": "",
            "train_start": pd.NaT,
            "train_end": pd.NaT,
            "test_start": pd.NaT,
            "test_end": pd.NaT,
            "status": "skipped_for_debug",
        }], columns=split_cols)
        tables["Train_Test_Splits"] = walk_splits
        tables["Regime_Train_Test_Splits"] = regime_splits
        tables["Split_Validation"] = pd.DataFrame([{
            "check_name": "debug_skip_splits",
            "ok": True,
            "violations": 0,
            "example": "",
            "status": "skipped_for_debug",
        }])
    else:
        walk_splits = build_full_walk_forward_splits(split_source, config.market)
        regime_splits = build_regime_train_test_splits(split_source, config.market)
        tables["Train_Test_Splits"] = walk_splits
        tables["Regime_Train_Test_Splits"] = regime_splits
        chrono_ok_wf, chrono_bad_wf = validate_chronological_splits(walk_splits)
        overlap_ok_wf, overlap_bad_wf = validate_no_train_test_overlap(walk_splits)
        chrono_ok_reg, chrono_bad_reg = validate_chronological_splits(regime_splits)
        overlap_ok_reg, overlap_bad_reg = validate_no_train_test_overlap(regime_splits)
        tables["Split_Validation"] = pd.DataFrame([
            {"check_name": "train_test_chronological_order", "ok": chrono_ok_wf, "violations": len(chrono_bad_wf), "example": str(chrono_bad_wf[:1])},
            {"check_name": "train_test_no_overlap", "ok": overlap_ok_wf, "violations": len(overlap_bad_wf), "example": str(overlap_bad_wf[:1])},
            {"check_name": "regime_train_test_chronological_order", "ok": chrono_ok_reg, "violations": len(chrono_bad_reg), "example": str(chrono_bad_reg[:1])},
            {"check_name": "regime_train_test_no_overlap", "ok": overlap_ok_reg, "violations": len(overlap_bad_reg), "example": str(overlap_bad_reg[:1])},
        ])
    if debug_skip_splits:
        placeholder_cols = ["split_id", "split_type", "regime_name", "regime_label", "train_start", "train_end", "test_start", "test_end", "status"]
        placeholder = pd.DataFrame([{
            "split_id": "",
            "split_type": "",
            "regime_name": "",
            "regime_label": "",
            "train_start": pd.NaT,
            "train_end": pd.NaT,
            "test_start": pd.NaT,
            "test_end": pd.NaT,
            "status": "skipped_for_debug",
        }], columns=placeholder_cols)
        tables["Gamma_R_Calibration_By_Split"] = placeholder.copy()
        tables["Gamma_R_Calibration_By_Regime_Split"] = placeholder.copy()
        oos = placeholder.copy()
        reg_oos = placeholder.copy()
        by_split = placeholder.copy()
        by_regime = placeholder.copy()
        backtest_status = "skipped_for_debug"
        regime_backtest_status = "skipped_for_debug"
    else:
        tables["Gamma_R_Calibration_By_Split"] = gamma_calibration_by_split(
            walk_splits, decisions, tables["Accepted_Profiles"], tables["BM_Exposure_Diagnostics"], config.market["hedge_intensity_scenarios"]
        )
        tables["Gamma_R_Calibration_By_Regime_Split"] = gamma_calibration_by_split(
            regime_splits, decisions, tables["Accepted_Profiles"], tables["BM_Exposure_Diagnostics"], config.market["hedge_intensity_scenarios"]
        )
        oos, by_split, _, _, backtest_status = run_split_backtests(
            tables["Stage_1_5_Handoff"],
            snapshot,
            valuation_date,
            config.market["hedge_intensity_scenarios"],
            gamma_R,
            tables["Gamma_R_Calibration_By_Split"],
            walk_splits,
            config.market.get("require_realized_future_spot", False),
            split_workflow_label="walk_forward",
        )
        reg_oos, _, _, by_regime, regime_backtest_status = run_split_backtests(
            tables["Stage_1_5_Handoff"],
            snapshot,
            valuation_date,
            config.market["hedge_intensity_scenarios"],
            gamma_R,
            tables["Gamma_R_Calibration_By_Regime_Split"],
            regime_splits,
            config.market.get("require_realized_future_spot", False),
            split_workflow_label="regime",
        )
    tables["Out_Of_Sample_Backtest"] = oos
    tables["Regime_Out_Of_Sample_Backtest"] = reg_oos
    tables["Backtest_By_Split"] = by_split
    tables["Backtest_By_Regime"] = by_regime if isinstance(by_regime, pd.DataFrame) else pd.DataFrame()
    if isinstance(by_regime, pd.DataFrame) and not by_regime.empty and "status" not in by_regime.columns:
        tables["Regime_Robustness_Summary"] = by_regime[[
            "regime_name", "hedge_intensity_scenario", "mean_pnl", "hit_rate",
            "mean_HE_t", "variance_reduction", "worst_case", "comment",
        ]]
    else:
        tables["Regime_Robustness_Summary"] = by_regime
    tables["Out_Of_Sample_Backtest_Status"] = pd.DataFrame([{"status": backtest_status}])
    tables["Regime_Backtest_Status"] = pd.DataFrame([{"status": regime_backtest_status}])
    tables["Sensitivity_Summary"] = sensitivity_summary(config.model, config.market, gamma_detail)
    tables["Aggregate_Hedge_Profile"] = aggregate_hedge_profile(decisions)
    tables["Cohort_Analysis"] = cohort_analysis(decisions, tables["Stage_1_5_Handoff"])
    if config.market.get("require_true_oos_backtest", True) and not debug_skip_splits and not fast_debug:
        split_oos, gamma_by_split_new = compute_oos_market_performance(
            forward_backtest_long=forward_backtest_long,
            spot_history_long=history,
            accepted_profiles=tables["Accepted_Profiles"],
            hedge_scenarios=config.market["hedge_intensity_scenarios"],
            stress_scenarios=stress_scenarios,
            vol_window_days=config.market["vol_window_days"],
            walk_forward_initial_train_start=config.market.get("walk_forward_initial_train_start", "2008-01-01"),
            walk_forward_initial_train_end=config.market.get("walk_forward_initial_train_end", "2013-12-31"),
            run_id=run_id,
        )
        tables["Split_OOS_Performance"] = split_oos
        tables["Gamma_R_Calibration_By_Split"] = gamma_by_split_new
    if fast_debug:
        tables["Profile_Cohort_Attribution"] = pd.DataFrame([{"run_id": run_id, "status": "skipped_for_debug"}])
        tables["Rolling_Market_Summary"] = pd.DataFrame([{"run_id": run_id, "status": "skipped_for_debug"}])
        tables["Regime_Performance"] = pd.DataFrame([{"run_id": run_id, "status": "skipped_for_debug"}])
        tables["Strategy_Ranking"] = pd.DataFrame([{"run_id": run_id, "status": "skipped_for_debug"}])
        tables["Negative_HE_Diagnostics"] = pd.DataFrame([{"run_id": run_id, "status": "skipped_for_debug"}])
    else:
        tables["Profile_Cohort_Attribution"] = compute_profile_cohort_attribution(
            rolling_market_performance=rolling_market,
            accepted_profiles=tables["Accepted_Profiles"],
            stage_1_5_handoff=tables["Stage_1_5_Handoff"],
            profile_sample_size=config.market.get("cohort_profile_sample_size", 1000),
            seed=config.market.get("cohort_sample_seed", 42),
            run_id=run_id,
        )
        tables["Rolling_Market_Summary"] = compute_rolling_market_summary(rolling_market)
        tables["Regime_Performance"] = compute_regime_performance(rolling_market)
        tables["Strategy_Ranking"] = compute_strategy_ranking(rolling_market)
        tables["Negative_HE_Diagnostics"] = compute_negative_he_diagnostics(rolling_market)
    tables = enrich_reporting_tables(tables, config)
    ckpt("pre-validation")
    validation_market = dict(config.market)
    if fast_debug:
        validation_market["debug_skip_optional_heavy"] = True
    v1 = validate_stage1(tables, config.model, config.run)
    v2 = validate_stage2(tables, validation_market, output_dir=run_output_dir, export_csv=False)
    tables["Validation_Checks"] = pd.concat([v1, v2], ignore_index=True)
    workbook_name = f"{Path(config.run['excel_workbook']).stem}_{run_id}.xlsx"
    debug_fast_export = bool(config.run.get("debug_fast_export", False) or config.market.get("debug_fast_export", False))
    csv_include_tables = None
    if debug_fast_export:
        csv_include_tables = {
            "Accepted_Profiles",
            "Inactive_Profiles",
            "BM_Exposure_Diagnostics",
            "Stage_1_5_Handoff",
            "Stage_2_Decisions",
            "Backtest_Results",
            "Validation_Checks",
            "Market_Data_Snapshot",
            "Rolling_Market_Summary",
            "Regime_Performance",
            "Strategy_Ranking",
            "Aggregate_Hedge_Profile",
            "Cohort_Analysis",
            "Family_Profile_Summary",
            "Family_Handoff_Summary",
            "Family_Stage2_Summary",
            "Output_Manifest",
        }
        if bool(config.run.get("debug_export_include_rolling_stage2_csv", False) or config.market.get("debug_export_include_rolling_stage2_csv", False)):
            csv_include_tables.add("Rolling_Stage_2_Decisions")
    workbook = export_tables(
        tables,
        config,
        stage="full",
        run_id=run_id,
        output_dir_override=run_output_dir,
        workbook_name_override=workbook_name,
        write_csv=True,
        write_workbook=False,
        csv_include_tables=csv_include_tables,
    )
    ckpt("csv export phase complete")
    # Recompute validation once CSV exports exist so file-level checks are auditable.
    v1 = validate_stage1(tables, config.model, config.run)
    v2 = validate_stage2(tables, validation_market, output_dir=run_output_dir, export_csv=config.run.get("export_csv", True))
    tables["Validation_Checks"] = pd.concat([v1, v2], ignore_index=True)
    ckpt("post-validation complete")
    workbook = export_tables(
        tables,
        config,
        stage="full",
        run_id=run_id,
        output_dir_override=run_output_dir,
        workbook_name_override=workbook_name,
        write_csv=False,
        write_workbook=True,
    )
    ckpt("workbook export phase complete")
    warnings = tables["Validation_Checks"][
        (tables["Validation_Checks"]["result"] == "fail") & (tables["Validation_Checks"]["severity"] == "warning")
    ]
    if not warnings.empty:
        print("Validation warnings: " + ", ".join(warnings["check_name"].head(10).tolist()))
    hard = hard_failures(tables["Validation_Checks"])
    if not hard.empty:
        names = ", ".join(hard["check_name"].head(10).tolist())
        raise ValueError(f"Validation hard failures after export: {names}")
    tables["Run_Metadata"] = pd.DataFrame([{"run_id": run_id, "output_dir": str(run_output_dir), "workbook_path": str(workbook)}])
    return tables, workbook


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stage2_only", "full_regen"], default="stage2_only")
    parser.add_argument("--reuse-stage1-dir", default=None)
    parser.add_argument("--reuse-run-id", default=None)
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--debug-skip-splits", action="store_true")
    args = parser.parse_args()
    _, path = run_full(
        mode=args.mode,
        reuse_stage1_dir=args.reuse_stage1_dir,
        reuse_run_id=args.reuse_run_id,
        no_clean=args.no_clean,
        debug_skip_splits=args.debug_skip_splits,
    )
    print(f"Full pipeline complete. Workbook: {path}")
