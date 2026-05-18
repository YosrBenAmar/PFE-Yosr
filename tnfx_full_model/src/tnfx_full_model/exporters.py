from __future__ import annotations

from datetime import datetime
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter


SHEET_ORDER = [
    "README_Run", "Config_Summary", "Family_Profile_Summary", "Family_Handoff_Summary",
    "Family_Stage2_Summary", "Accepted_Profiles", "Inactive_Profiles",
    "BM_Exposure_Diagnostics", "Timing_CV_Scenarios", "Tenor_Weights",
    "Tenor_Weights_Preview", "Stage_1_5_Handoff", "Stage_1_5_Handoff_Preview",
    "Macro_Anchor_Check", "Rejection_Log", "Sobol_Acceptance", "Sobol_Meta",
    "Sign_Threshold_Sensitivity", "Regime_State_Priors", "Market_Load_Metadata",
    "Forward_Backtest_Long", "Rolling_Market_Performance", "Split_OOS_Performance",
    "Train_Test_Splits", "Gamma_R_Calibration_By_Split", "Out_Of_Sample_Backtest",
    "Backtest_By_Split", "Regime_Train_Test_Splits", "Gamma_R_Calibration_By_Regime_Split",
    "Regime_Out_Of_Sample_Backtest", "Backtest_By_Regime", "Regime_Robustness_Summary",
    "Methodology_Status", "Output_Manifest",
    "Market_Data_Snapshot", "Stage_2_Decisions", "Stage_2_Decisions_Preview",
    "Backtest_Results", "Sensitivity_Summary", "Aggregate_Hedge_Profile",
    "Cohort_Analysis", "Rolling_Market_Summary", "Regime_Performance",
    "Strategy_Ranking", "Negative_HE_Diagnostics", "Profile_Cohort_Attribution",
    "Validation_Checks",
]


def config_summary(config, run_id: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [{"section": "run", "key": k, "value": v} for k, v in config.run.items()]
        + [{"section": "market", "key": k, "value": v} for k, v in config.market.items()]
        + ([{"section": "runtime", "key": "run_id", "value": run_id}] if run_id else [])
    )


def _stratified_preview(df: pd.DataFrame, max_rows: int, family_col: str = "family") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if family_col not in df.columns:
        return df.head(max_rows).copy()
    families = [f for f in ["importer", "exporter", "processor", "trader"] if f in set(df[family_col].dropna().unique())]
    if not families:
        families = list(df[family_col].dropna().astype(str).unique())
    grouped = {fam: df[df[family_col] == fam] for fam in families}
    alloc = {fam: 0 for fam in families}
    remaining = int(max_rows)
    cursor = 0
    while remaining > 0:
        candidates = [fam for fam in families if alloc[fam] < len(grouped[fam])]
        if not candidates:
            break
        fam = candidates[cursor % len(candidates)]
        alloc[fam] += 1
        remaining -= 1
        cursor += 1
    parts = [grouped[fam].head(alloc[fam]) for fam in families if alloc[fam] > 0]
    if not parts:
        return df.head(max_rows).copy()
    return pd.concat(parts, ignore_index=True)


def _family_profile_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    accepted = tables.get("Accepted_Profiles", pd.DataFrame())
    inactive = tables.get("Inactive_Profiles", pd.DataFrame())
    diag = tables.get("BM_Exposure_Diagnostics", pd.DataFrame())
    if accepted.empty:
        return pd.DataFrame(columns=[
            "family", "accepted_profiles", "inactive_profiles", "accepted_share",
            "target_prior_share", "mean_h_R", "mean_h_C", "mean_r", "mean_beta",
            "mean_fx_debt_service_share", "mean_delta_CF_total", "mean_delta_profit_total",
            "mean_delta_net_EUR", "mean_delta_net_USD", "mean_currency_mismatch_gap",
        ])
    prof = accepted.merge(
        diag[["profile_id", "delta_CF_total", "delta_profit_total", "delta_net_EUR", "delta_net_USD", "currency_mismatch_gap"]],
        on="profile_id",
        how="left",
    )
    accepted_counts = prof.groupby("family").size().rename("accepted_profiles")
    inactive_counts = inactive.groupby("family").size().rename("inactive_profiles") if not inactive.empty else pd.Series(dtype=float)
    summary = prof.groupby("family", as_index=False).agg(
        mean_h_R=("h_R", "mean"),
        mean_h_C=("h_C", "mean"),
        mean_r=("r", "mean"),
        mean_beta=("beta", "mean"),
        mean_fx_debt_service_share=("fx_debt_service_share", "mean"),
        mean_delta_CF_total=("delta_CF_total", "mean"),
        mean_delta_profit_total=("delta_profit_total", "mean"),
        mean_delta_net_EUR=("delta_net_EUR", "mean"),
        mean_delta_net_USD=("delta_net_USD", "mean"),
        mean_currency_mismatch_gap=("currency_mismatch_gap", "mean"),
        target_prior_share=("target_prior_family_share", "mean"),
    )
    summary["accepted_profiles"] = summary["family"].map(accepted_counts).fillna(0).astype(int)
    summary["inactive_profiles"] = summary["family"].map(inactive_counts).fillna(0).astype(int)
    total = max(int(summary["accepted_profiles"].sum()), 1)
    summary["accepted_share"] = summary["accepted_profiles"] / total
    return summary[
        [
            "family", "accepted_profiles", "inactive_profiles", "accepted_share",
            "target_prior_share", "mean_h_R", "mean_h_C", "mean_r", "mean_beta",
            "mean_fx_debt_service_share", "mean_delta_CF_total", "mean_delta_profit_total",
            "mean_delta_net_EUR", "mean_delta_net_USD", "mean_currency_mismatch_gap",
        ]
    ]


def _family_handoff_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    handoff = tables.get("Stage_1_5_Handoff", pd.DataFrame())
    if handoff.empty:
        return pd.DataFrame(columns=[
            "family", "handoff_rows", "active_profiles", "profiles_with_one_currency",
            "profiles_with_two_currencies", "mean_E_t", "total_Q_plus", "total_Q_minus",
            "dominant_currency",
        ])
    counts = handoff.groupby(["family", "profile_id"])["currency"].nunique().reset_index(name="n_currency")
    one = counts[counts["n_currency"] == 1].groupby("family").size()
    two = counts[counts["n_currency"] >= 2].groupby("family").size()
    dom = handoff.groupby(["family", "currency"])["E_t"].apply(lambda s: s.abs().sum()).reset_index()
    dom = dom.loc[dom.groupby("family")["E_t"].idxmax()][["family", "currency"]].rename(columns={"currency": "dominant_currency"})
    summary = handoff.groupby("family", as_index=False).agg(
        handoff_rows=("profile_id", "size"),
        active_profiles=("profile_id", "nunique"),
        mean_E_t=("E_t", "mean"),
        total_Q_plus=("Q_plus", "sum"),
        total_Q_minus=("Q_minus", "sum"),
    )
    summary["profiles_with_one_currency"] = summary["family"].map(one).fillna(0).astype(int)
    summary["profiles_with_two_currencies"] = summary["family"].map(two).fillna(0).astype(int)
    summary = summary.merge(dom, on="family", how="left")
    return summary[
        [
            "family", "handoff_rows", "active_profiles", "profiles_with_one_currency",
            "profiles_with_two_currencies", "mean_E_t", "total_Q_plus", "total_Q_minus",
            "dominant_currency",
        ]
    ]


def _family_stage2_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    dec = tables.get("Stage_2_Decisions", pd.DataFrame())
    if dec.empty:
        return pd.DataFrame(columns=[
            "family", "stage2_decision_rows", "material_rows", "immaterial_rows",
            "mean_h_c", "mean_expected_cost", "mean_HE_t_excluding_immaterial",
            "pct_lower_bound_binding", "pct_upper_bound_binding", "pct_immaterial",
        ])
    tmp = dec.copy()
    tmp["is_material"] = tmp["stage2_row_status"].fillna("material").eq("material")
    grouped = tmp.groupby("family", as_index=False).agg(
        stage2_decision_rows=("profile_id", "size"),
        material_rows=("is_material", "sum"),
        mean_h_c=("h_c", "mean"),
        mean_expected_cost=("expected_cost", "mean"),
    )
    grouped["material_rows"] = grouped["material_rows"].astype(int)
    grouped["immaterial_rows"] = grouped["stage2_decision_rows"] - grouped["material_rows"]
    he = tmp[tmp["is_material"]].groupby("family")["HE_t"].mean()
    grouped["mean_HE_t_excluding_immaterial"] = grouped["family"].map(he)
    lower = tmp.groupby("family")["binding_constraint"].apply(lambda s: (s == "lower").mean())
    upper = tmp.groupby("family")["binding_constraint"].apply(lambda s: (s == "upper").mean())
    imm = tmp.groupby("family")["binding_constraint"].apply(lambda s: (s == "immaterial").mean())
    grouped["pct_lower_bound_binding"] = grouped["family"].map(lower).fillna(0.0)
    grouped["pct_upper_bound_binding"] = grouped["family"].map(upper).fillna(0.0)
    grouped["pct_immaterial"] = grouped["family"].map(imm).fillna(0.0)
    return grouped[
        [
            "family", "stage2_decision_rows", "material_rows", "immaterial_rows",
            "mean_h_c", "mean_expected_cost", "mean_HE_t_excluding_immaterial",
            "pct_lower_bound_binding", "pct_upper_bound_binding", "pct_immaterial",
        ]
    ]


def enrich_reporting_tables(tables: dict[str, pd.DataFrame], config) -> dict[str, pd.DataFrame]:
    enriched = dict(tables)
    max_rows = int(config.run.get("max_excel_rows_per_sheet", 50_000))
    if "Tenor_Weights" in enriched and "Accepted_Profiles" in enriched and not enriched["Tenor_Weights"].empty:
        fam_map = enriched["Accepted_Profiles"][["profile_id", "family"]].drop_duplicates()
        tw = enriched["Tenor_Weights"].merge(fam_map, on="profile_id", how="left")
        enriched["Tenor_Weights_Preview"] = _stratified_preview(tw, max_rows)
    else:
        enriched["Tenor_Weights_Preview"] = pd.DataFrame()
    enriched["Stage_1_5_Handoff_Preview"] = _stratified_preview(enriched.get("Stage_1_5_Handoff", pd.DataFrame()), max_rows)
    enriched["Stage_2_Decisions_Preview"] = _stratified_preview(enriched.get("Stage_2_Decisions", pd.DataFrame()), max_rows)
    enriched["Family_Profile_Summary"] = _family_profile_summary(enriched)
    enriched["Family_Handoff_Summary"] = _family_handoff_summary(enriched)
    enriched["Family_Stage2_Summary"] = _family_stage2_summary(enriched)
    return enriched


def readme_run(config, truncated: dict[str, tuple[int, int]], stage: str, tables: dict[str, pd.DataFrame], run_id: str | None = None) -> pd.DataFrame:
    macro = tables.get("Macro_Anchor_Check", pd.DataFrame())
    macro_status = "" if macro.empty or "macro_anchor_status" not in macro.columns else macro["macro_anchor_status"].iloc[0]
    acceptance = tables.get("Sobol_Acceptance", pd.DataFrame())
    acceptance_status = "" if acceptance.empty else "; ".join(f"{r.family}:{r.status}" for r in acceptance.itertuples())
    meta = tables.get("Market_Load_Metadata", pd.DataFrame())
    workbook_loaded_status = "" if meta.empty or "workbook_loaded_status" not in meta.columns else meta["workbook_loaded_status"].iloc[0]
    backtest = tables.get("Backtest_Results", pd.DataFrame())
    backtest_status = ""
    if not backtest.empty and "status" in backtest.columns:
        backtest_status = str(backtest["status"].iloc[0])
    elif not backtest.empty:
        backtest_status = "backtest_completed"
    rows = [
        {"item": "run_id", "value": run_id or ""},
        {"item": "stage", "value": stage},
        {"item": "valuation_date", "value": config.run["valuation_date"]},
        {"item": "market_data_source", "value": config.market.get("market_data_source", "")},
        {"item": "market_workbook_path", "value": config.market.get("market_workbook_path", "")},
        {"item": "workbook_loaded_status", "value": workbook_loaded_status},
        {"item": "backtest_status", "value": backtest_status},
        {"item": "macro_anchor_status", "value": macro_status},
        {"item": "acceptance_status", "value": acceptance_status},
        {"item": "reproducibility", "value": f"seed={config.run['seed']}; sobol_dimension={config.run['sobol_dimension']}; burn_in={config.run['sobol_burn_in']}"},
        {"item": "Forwards_calculations.xlsx", "value": "Configured Stage 2 market-data workbook when market_data_source=excel_workbook."},
        {"item": "Brown-Toft quantity term", "value": "sigma_Q and rho are kept in the FOC; brown_toft_constant=0 by default; simplified variance implementation."},
        {"item": "preview_policy", "value": "Large tables are fully exported to CSV. Excel preview sheets are family-stratified so the workbook does not visually show only the first family."},
        {"item": "full_period_walk_forward_enabled", "value": "true"},
        {"item": "full_walk_forward_enabled", "value": "true"},
        {"item": "regime_aware_backtest_enabled", "value": str(bool(config.market.get("regime_backtest", {}).get("enabled", False))).lower()},
        {"item": "split_specific_gamma_R_used", "value": "false"},
        {"item": "out_of_sample_backtest_status", "value": "diagnostic_not_true_out_of_sample"},
        {"item": "true_oos_backtest_disclosure", "value": "Train/test split tables are diagnostic only unless split_specific_gamma_used=true. Current run does not claim true out-of-sample hedge recomputation."},
    ]
    splits = tables.get("Train_Test_Splits", pd.DataFrame())
    if isinstance(splits, pd.DataFrame) and not splits.empty:
        regimes = ", ".join(sorted(splits["regime_name"].dropna().astype(str).unique().tolist()))
        rows.append({"item": "number_of_regime_splits", "value": str(int((splits["split_type"] == "regime_train_test").sum()))})
        rows.append({"item": "regime_names", "value": regimes})
        sample = splits[["split_id", "train_start", "train_end", "test_start", "test_end"]].head(8).to_dict("records")
        rows.append({"item": "split_period_examples", "value": str(sample)})
        first = splits[splits["split_id"].astype(str).str.startswith("wf_")].sort_values("split_id").head(1)
        if not first.empty:
            r = first.iloc[0]
            rows.extend([
                {"item": "first_walk_forward_train_start", "value": str(r.get("train_start", ""))},
                {"item": "first_walk_forward_train_end", "value": str(r.get("train_end", ""))},
                {"item": "first_walk_forward_test_start", "value": str(r.get("test_start", ""))},
                {"item": "first_walk_forward_test_end", "value": str(r.get("test_end", ""))},
                {"item": "n_walk_forward_splits", "value": str(int((splits["split_id"].astype(str).str.startswith("wf_")).sum()))},
            ])
    regime_splits = tables.get("Regime_Train_Test_Splits", pd.DataFrame())
    if isinstance(regime_splits, pd.DataFrame) and not regime_splits.empty:
        rows.append({"item": "n_regime_splits", "value": str(len(regime_splits))})
    if "Regime_Backtest_Status" in tables and isinstance(tables["Regime_Backtest_Status"], pd.DataFrame) and not tables["Regime_Backtest_Status"].empty:
        rows.append({"item": "regime_backtest_status", "value": str(tables["Regime_Backtest_Status"]["status"].iloc[0])})
    if "Out_Of_Sample_Backtest_Status" in tables and isinstance(tables["Out_Of_Sample_Backtest_Status"], pd.DataFrame) and not tables["Out_Of_Sample_Backtest_Status"].empty:
        rows.append({"item": "realized_future_spot_status", "value": str(tables["Out_Of_Sample_Backtest_Status"]["status"].iloc[0])})
    for sheet, (written, total) in truncated.items():
        rows.append({"item": f"truncated_sheet:{sheet}", "value": f"Excel wrote {written} of {total} rows; CSV export contains full table when enabled."})
    return pd.DataFrame(rows)


def _ensure_run_id_column(df: pd.DataFrame, run_id: str | None) -> pd.DataFrame:
    if run_id is None:
        return df
    out = df.copy()
    if "run_id" not in out.columns:
        out.insert(0, "run_id", run_id)
    return out


def _methodology_status(tables: dict[str, pd.DataFrame], run_config: dict) -> pd.DataFrame:
    macro = tables.get("Macro_Anchor_Check", pd.DataFrame())
    macro_status = "unknown"
    if not macro.empty and "macro_anchor_status" in macro.columns:
        macro_status = str(macro["macro_anchor_status"].iloc[0])
    rolling = tables.get("Rolling_Market_Performance", pd.DataFrame())
    oos = tables.get("Split_OOS_Performance", pd.DataFrame())
    realized_status = "implemented_via_workbook_forwards" if isinstance(rolling, pd.DataFrame) and not rolling.empty else "not_implemented"
    realized_explanation = (
        "Rolling market-side backtest computed on ~116k historical forward observations from Forwards_calculations.xlsx spanning 2007-2026, across five hedge intensity scenarios and two CIP stress scenarios. See Rolling_Market_Performance.csv."
        if realized_status == "implemented_via_workbook_forwards"
        else "Rolling market-side backtest not computed in this run because Rolling_Market_Performance is missing or empty."
    )
    true_oos = isinstance(oos, pd.DataFrame) and not oos.empty and "split_specific_gamma_used" in oos.columns and bool(oos["split_specific_gamma_used"].astype(bool).any())
    return pd.DataFrame([
        {"item": "Stage 1 profile engine", "status": "implemented", "severity": "info", "explanation": ""},
        {"item": "Stage 1.5 currency-tenor handoff", "status": "implemented", "severity": "info", "explanation": ""},
        {"item": "Global Stage 2 hedge-decision engine", "status": "implemented", "severity": "info", "explanation": ""},
        {"item": "Historical realized backtest", "status": realized_status, "severity": "info", "explanation": realized_explanation},
        {"item": "True train/test out-of-sample backtest", "status": "implemented" if true_oos else "not_implemented", "severity": "info" if true_oos else ("warning" if not run_config.get("require_true_oos_backtest", False) else "hard"), "explanation": "Walk-forward expanding-window splits with split-specific gamma_R recalibration on training data only. Decisions recomputed per split, not filtered from global decisions. See Split_OOS_Performance.csv and Gamma_R_Calibration_By_Split.csv." if true_oos else "Current split tables are diagnostic unless split_specific_gamma_used=true."},
        {"item": "CIP+50bps stress scenario", "status": "implemented", "severity": "info", "explanation": "CIP+50bps consumed by optimizer as a parallel forward_stress_scenario. Compare with cip_base in Rolling_Market_Performance.csv via groupby forward_stress_scenario."},
        {"item": "Per-profile time-series P&L", "status": "not_produced_by_design", "severity": "info", "explanation": "Synthetic profiles are static Sobol draws and have no temporal identity. Per-profile drilldown is on-demand only via the Streamlit interface (Session 2)."},
        {"item": "Macro-anchor", "status": macro_status, "severity": "warning" if macro_status == "nominal_pass_only_prior_sensitive" else "info", "explanation": "Primary calibration uses nominal prior. Firm-count and trade-flow priors are sensitivity views."},
        {"item": "GARCH volatility", "status": "future_placeholder", "severity": "info", "explanation": ""},
        {"item": "CVaR objective", "status": "future_placeholder", "severity": "info", "explanation": ""},
        {"item": "Options", "status": "future_stage_2_5", "severity": "info", "explanation": ""},
        {"item": "Stage 3 CIP basis", "status": "future_work", "severity": "info", "explanation": ""},
    ])


def export_tables(
    tables: dict[str, pd.DataFrame],
    config,
    stage: str = "full",
    run_id: str | None = None,
    output_dir_override: Path | None = None,
    workbook_name_override: str | None = None,
    write_csv: bool = True,
    write_workbook: bool = True,
    csv_include_tables: set[str] | None = None,
) -> Path:
    out_dir = output_dir_override or config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    export_csv = bool(config.run.get("export_csv", True)) and write_csv
    max_rows = int(config.run.get("max_excel_rows_per_sheet", 1_000_000))
    workbook_name = workbook_name_override or config.run["excel_workbook"]
    workbook = out_dir / workbook_name
    truncated = {}
    tables_local = {k: (_ensure_run_id_column(v, run_id) if isinstance(v, pd.DataFrame) else v) for k, v in tables.items()}
    if "Methodology_Status" not in tables_local:
        tables_local["Methodology_Status"] = _ensure_run_id_column(_methodology_status(tables_local, config.run), run_id)
    manifest_rows = []
    if export_csv:
        for name, df in tables_local.items():
            if not isinstance(df, pd.DataFrame):
                continue
            if csv_include_tables is not None and name not in csv_include_tables:
                continue
            path = out_dir / f"{name}.csv"
            temp_path = out_dir / f".{name}.csv.tmp"
            status = "exported"
            error_message = ""
            for attempt in range(1, 5):
                try:
                    df.to_csv(temp_path, index=False)
                    temp_path.replace(path)
                    break
                except (PermissionError, OSError) as exc:
                    status = "failed"
                    error_message = str(exc)
                    if temp_path.exists():
                        try:
                            temp_path.unlink()
                        except OSError:
                            pass
                    if attempt == 4:
                        break
                    time.sleep(0.5 * attempt)
            manifest_rows.append({
                "run_id": run_id or "",
                "table_name": name,
                "file_name": path.name,
                "file_path": str(path),
                "n_rows": int(len(df)),
                "n_columns": int(len(df.columns)),
                "export_status": status,
                "export_timestamp": datetime.now().isoformat(timespec="seconds"),
                "error_message": error_message,
            })
    excel_tables = dict(tables_local)
    excel_tables["Config_Summary"] = _ensure_run_id_column(config_summary(config, run_id), run_id)
    large_preview_map = {
        "Tenor_Weights": "Tenor_Weights_Preview",
        "Stage_1_5_Handoff": "Stage_1_5_Handoff_Preview",
        "Stage_2_Decisions": "Stage_2_Decisions_Preview",
    }
    for name, df in list(excel_tables.items()):
        if not isinstance(df, pd.DataFrame):
            continue
        if len(df) > max_rows:
            truncated[name] = (max_rows, len(df))
            preview_name = large_preview_map.get(name)
            if preview_name and preview_name in excel_tables and not excel_tables[preview_name].empty:
                excel_tables[name] = excel_tables[preview_name].head(max_rows)
            else:
                excel_tables[name] = _stratified_preview(df, max_rows)
    if manifest_rows:
        excel_tables["Output_Manifest"] = pd.DataFrame(manifest_rows)
        if export_csv:
            excel_tables["Output_Manifest"].to_csv(out_dir / "Output_Manifest.csv", index=False)
    else:
        excel_tables["Output_Manifest"] = pd.DataFrame(columns=["run_id", "table_name", "file_name", "file_path", "n_rows", "n_columns", "export_status", "export_timestamp", "error_message"])
    excel_tables["README_Run"] = _ensure_run_id_column(readme_run(config, truncated, stage, tables_local, run_id), run_id)

    def _write_excel(target: Path) -> None:
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            for sheet in SHEET_ORDER:
                if sheet in excel_tables:
                    excel_tables[sheet].to_excel(writer, sheet_name=sheet[:31], index=False)
            wb = writer.book
            for ws in wb.worksheets:
                ws.freeze_panes = "A2"
                ws.auto_filter.ref = ws.dimensions
                for idx, col in enumerate(ws.columns, start=1):
                    sample_cells = list(col[:1000])
                    width = min(max(len(str(cell.value)) if cell.value is not None else 0 for cell in sample_cells) + 2, 40)
                    ws.column_dimensions[get_column_letter(idx)].width = width

    if not write_workbook:
        return workbook
    try:
        _write_excel(workbook)
        latest = config.root / "data" / "outputs" / "tnfx_full_model_results_latest.xlsx"
        shutil.copyfile(workbook, latest)
        return workbook
    except PermissionError:
        time.sleep(0.5)
        try:
            _write_excel(workbook)
            latest = config.root / "data" / "outputs" / "tnfx_full_model_results_latest.xlsx"
            shutil.copyfile(workbook, latest)
            return workbook
        except PermissionError:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback = workbook.with_name(f"{workbook.stem}_{stamp}{workbook.suffix}")
            _write_excel(fallback)
            latest = config.root / "data" / "outputs" / "tnfx_full_model_results_latest.xlsx"
            shutil.copyfile(fallback, latest)
            return fallback
