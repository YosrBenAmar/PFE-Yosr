from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tnfx_full_model.config_loader import load_project_config
from tnfx_full_model.exporters import enrich_reporting_tables, export_tables
from tnfx_full_model.regime_engine import all_regime_priors
from tnfx_full_model.sensitivity import macro_anchor_check, sign_threshold_sensitivity
from tnfx_full_model.sobol_sampler import sample_profiles
from tnfx_full_model.stage15_handoff import build_handoff
from tnfx_full_model.timing_kernel import build_tenor_weights, timing_scenarios
from tnfx_full_model.validation import hard_failures, validate_stage1


def run_stage1(run_id: str | None = None):
    config = load_project_config(ROOT)
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = config.root / "data" / "outputs" / run_id
    sampled = sample_profiles(config.model, config.run, config.run["primary_prior_mode"])
    tables = dict(sampled)
    tables["Timing_CV_Scenarios"] = timing_scenarios(config.model)
    tables["Regime_State_Priors"] = pd.DataFrame(all_regime_priors(config.model))
    tables["Tenor_Weights"] = build_tenor_weights(tables["Accepted_Profiles"], config.model)
    tables["Stage_1_5_Handoff"] = build_handoff(
        tables["Accepted_Profiles"],
        tables["BM_Exposure_Diagnostics"],
        tables["Tenor_Weights"],
        config.model["active_currency_threshold"],
    )
    tables["Macro_Anchor_Check"] = macro_anchor_check(tables["BM_Exposure_Diagnostics"], config.model)
    tables["Sign_Threshold_Sensitivity"] = sign_threshold_sensitivity(tables["BM_Exposure_Diagnostics"], config.model)
    tables = enrich_reporting_tables(tables, config)
    tables["Validation_Checks"] = validate_stage1(tables, config.model, config.run)
    workbook_name = f"{Path(config.run['excel_workbook']).stem}_{run_id}.xlsx"
    workbook = export_tables(
        tables,
        config,
        stage="stage1",
        run_id=run_id,
        output_dir_override=run_output_dir,
        workbook_name_override=workbook_name,
    )
    warnings = tables["Validation_Checks"][
        (tables["Validation_Checks"]["result"] == "fail") & (tables["Validation_Checks"]["severity"] == "warning")
    ]
    if not warnings.empty:
        print("Stage 1 validation warnings: " + ", ".join(warnings["check_name"].head(10).tolist()))
    hard = hard_failures(tables["Validation_Checks"])
    if not hard.empty:
        names = ", ".join(hard["check_name"].head(10).tolist())
        raise SystemExit(f"Stage 1 validation hard failures after export: {names}")
    tables["Run_Metadata"] = pd.DataFrame([{"run_id": run_id, "output_dir": str(run_output_dir), "workbook_path": str(workbook)}])
    return tables, workbook


if __name__ == "__main__":
    _, path = run_stage1()
    print(f"Stage 1 complete. Workbook: {path}")
