from pathlib import Path

import pandas as pd

from tnfx_full_model.exporters import enrich_reporting_tables, export_tables


def test_output_manifest_has_run_id(config, tmp_path):
    run_id = "20990101_010101"
    tables = {
        "Accepted_Profiles": pd.DataFrame([{"profile_id": 1, "family": "importer", "h_R": 0.1, "h_C": 0.2, "r": 0.1, "beta": 0.8, "fx_debt_service_share": 0.0, "target_prior_family_share": 0.35}]),
        "Inactive_Profiles": pd.DataFrame(columns=["profile_id", "family"]),
        "BM_Exposure_Diagnostics": pd.DataFrame([{"profile_id": 1, "family": "importer", "delta_CF_total": -0.08, "delta_profit_total": -0.8, "delta_net_EUR": -0.02, "delta_net_USD": -0.01, "currency_mismatch_gap": 0.0}]),
        "Stage_1_5_Handoff": pd.DataFrame(columns=["profile_id", "family", "currency", "E_t", "Q_plus", "Q_minus", "delta_net_k", "timing_cv_scenario", "currency_pair", "tenor_months"]),
        "Stage_2_Decisions": pd.DataFrame(columns=["profile_id", "family", "h_c", "lambda", "direction", "pricing_side"]),
        "Macro_Anchor_Check": pd.DataFrame([{"prior_mode": "nominal", "pass_fail": "pass", "macro_anchor_status": "all_priors_pass"}]),
        "Sobol_Acceptance": pd.DataFrame([{"family": "importer", "status": "ok"}]),
        "Backtest_Results": pd.DataFrame([{"status": "backtest_skipped_missing_realized_future_spot"}]),
    }
    tables = enrich_reporting_tables(tables, config)
    workbook = export_tables(
        tables,
        config,
        stage="full",
        run_id=run_id,
        output_dir_override=Path(tmp_path),
        workbook_name_override=f"unit_{run_id}.xlsx",
    )
    manifest_path = Path(tmp_path) / "Output_Manifest.csv"
    assert workbook.exists()
    assert manifest_path.exists()
    manifest = pd.read_csv(manifest_path)
    assert not manifest.empty
    assert set(manifest["run_id"].astype(str).unique()) == {run_id}

