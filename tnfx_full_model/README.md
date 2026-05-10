# TNFX Full Model

Reproducible Python project for Tunisian corporate FX exposure profiles and forward hedging decisions.

The project has three implemented layers:

- Stage 1: Sobol synthetic Tunisian corporate FX profile generation.
- Stage 1.5: Currency-aware tenor exposure handoff rows.
- Stage 2: Market-data join, CIP forwards, rolling volatility, hedge-intensity calibration, hedge decisions, and realized forward P&L backtest.

Not implemented by design: options, Stage 2.5, empirical Stage 3 CIP-basis modeling, PDF reporting, and CVaR optimization. CVaR appears only as a future-placeholder label in sensitivity output.

## Commands

```powershell
pip install -r requirements.txt
python pipeline_stage1.py
python pipeline_full.py
pytest
```

## Methodology Choices Closed In Code

`beta = 1 - pass_through` is treated as margin absorption. Operating effective exposure is `delta_op_eff_k = (1 - beta) * delta_CF_k`.

The Sobol architecture is exactly 15-dimensional. The first 1,024 points are skipped. A 1,000-point pilot estimates empirical acceptance by family, then the candidate draw size is `ceil(N_target / alpha * 1.10)`. Acceptance thresholds are:

- `alpha < 0.30`: hard fail.
- `alpha < 0.50`: validation fail/manual recalibration required.
- `alpha < 0.65`: warning.

Regime states are binary over `g_NR, g_PAE, g_TE, g_AEO, g_ACC, g_EXT, g_CIRC`. Feasible states are enumerated completely by `src/tnfx_full_model/regime_engine.py` from these rules:

- `g_TE <= g_PAE`
- `g_CIRC=1` only for importer, processor, trader
- `g_CIRC=0` for exporter
- `g_PAE=1` is not allowed for importer
- `g_NR=1` implies `g_PAE=1`
- `g_PAE=1` implies `g_ACC=1`

Default regime probabilities are author-calibrated. They are generated transparently from feasible states by multiplying the score weights in `config/model_config.yaml` and normalizing within family. This provides the complete feasible list at runtime without hidden hand curation.

Regime bound effects are also in `config/model_config.yaml`. If a flag has no direct effect on a variable, its shift is zero. The affected variables are `h_R`, `beta`, `c`, `lambda`, `sigma_Q`, and `fx_debt_service_share`; all adjusted values are clipped back into family bounds, and lambda is clipped to 0.95.

`pipeline_stage1.py` never needs market data. `pipeline_full.py` stops with a clear missing-file or missing-column error if market or spot-history inputs are incomplete.

`Forwards calculations.xlsx` is the default Stage 2 market-data source when `market_data_source: excel_workbook`. Templates are documentation scaffolds only and are not runtime inputs.

The Brown-Toft quantity convention keeps `sigma_Q` and `rho` in the hedge FOC, sets `brown_toft_constant = 0` by default, and documents the variance engine as a simplified implementation. No extra quantity-risk terms are invented.

Excel export writes every CSV when `export_csv = true`. Excel sheets may be truncated to `max_excel_rows_per_sheet`; any truncation is recorded in `README_Run`.
