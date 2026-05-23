"""
run_external_validation.py - entry point for the external validation layer.

Run from the project root:

    python run_external_validation.py

It expects:
    data/external/colombus_transactions.xlsx
    data/outputs/latest/*.csv

It writes:
    data/outputs/latest/Validation_*.csv
"""

from pathlib import Path

from external_validation import run_external_validation


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = run_external_validation(
        colombus_path=root / "data" / "external" / "colombus_transactions.xlsx",
        outputs_dir=root / "data" / "outputs" / "latest",
        verbose=True,
    )
    print("\n=== EXTERNAL VALIDATION SUMMARY ===")
    for test_name, summary in result.items():
        print(f"\n{test_name}: {summary.get('verdict', 'see CSV')}")
        for k, v in summary.items():
            if k == "verdict":
                continue
            print(f"  {k}: {v}")

