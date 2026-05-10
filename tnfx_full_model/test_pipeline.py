#!/usr/bin/env python
"""Test script to verify patch functions work and generate expected tables."""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline_full import run_full

# Run the full pipeline
try:
    with open("pipeline_log.txt", "w") as log_file:
        log_file.write("Starting pipeline...\n")
        log_file.flush()
        
        print("Starting pipeline...")
    tables, workbook = run_full()
    
    print(f"\n✓ Pipeline completed successfully!")
    print(f"✓ Output workbook: {workbook}")
    print(f"\n=== Generated Tables ({len(tables)} total) ===")
    
    # Check for new patch tables
    new_tables = [
        "Forward_Backtest_Long",
        "Rolling_Market_Performance", 
        "Split_OOS_Performance",
        "Profile_Cohort_Attribution",
        "Rolling_Market_Summary",
        "Regime_Performance",
        "Strategy_Ranking",
        "Negative_HE_Diagnostics",
        "Gamma_R_Calibration_By_Split",
    ]
    
    for table_name in sorted(tables.keys()):
        df = tables[table_name]
        is_new = table_name in new_tables
        marker = "⭐ NEW" if is_new else "     "
        size_str = f"{len(df):,} rows"
        print(f"{marker}  {table_name}: {size_str}")
        
    print(f"\n=== New Patch Table Details ===")
    for table_name in new_tables:
        if table_name in tables:
            df = tables[table_name]
            print(f"\n{table_name}:")
            print(f"  Rows: {len(df):,}")
            print(f"  Columns: {list(df.columns)[:5]}..." if len(df.columns) > 5 else f"  Columns: {list(df.columns)}")
        else:
            print(f"\n{table_name}: ❌ NOT GENERATED")
    
    # Check run_id consistency
    run_ids = set()
    for df in tables.values():
        if isinstance(df, pd.DataFrame) and "run_id" in df.columns:
            run_ids.update(df["run_id"].dropna().unique())
    
    if run_ids:
        print(f"\n=== Run ID ===")
        print(f"Run IDs found: {run_ids}")
        if len(run_ids) == 1:
            print(f"✓ All tables share single run_id")
        else:
            print(f"⚠ Multiple run_ids detected: {run_ids}")

except Exception as e:
    print(f"❌ Pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
