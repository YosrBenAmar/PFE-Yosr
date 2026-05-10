from __future__ import annotations

import pandas as pd


def _date_col(df: pd.DataFrame) -> str:
    if "valuation_date" in df.columns:
        return "valuation_date"
    if "date" in df.columns:
        return "date"
    raise ValueError("No date column found for split construction.")


def _to_ts(value) -> pd.Timestamp:
    return pd.to_datetime(value)


def build_full_walk_forward_splits(market_data: pd.DataFrame, config: dict) -> pd.DataFrame:
    cols = [
        "split_id", "split_type", "regime_name", "regime_label",
        "train_start", "train_end", "test_start", "test_end",
        "expanding_window", "split_status", "n_train_market_rows", "n_test_market_rows",
    ]
    split_cfg = config.get("backtest_split", {})
    if not split_cfg:
        split_cfg = {
            "mode": "walk_forward",
            "walk_forward_initial_train_start": "2008-01-01",
            "walk_forward_initial_train_end": "2013-12-31",
            "walk_forward_test_frequency": "yearly",
            "expanding_window": True,
            "require_walk_forward_backtest": False,
        }
    if market_data.empty:
        return pd.DataFrame([{
            "split_id": "wf_000",
            "split_type": "full_walk_forward",
            "regime_name": "full_period_walk_forward",
            "regime_label": "Expanding-window full-period walk-forward",
            "train_start": pd.to_datetime(split_cfg["walk_forward_initial_train_start"]),
            "train_end": pd.to_datetime(split_cfg["walk_forward_initial_train_end"]),
            "test_start": pd.NaT,
            "test_end": pd.NaT,
            "expanding_window": bool(split_cfg.get("expanding_window", True)),
            "split_status": "insufficient_initial_training_window",
            "n_train_market_rows": 0,
            "n_test_market_rows": 0,
        }], columns=cols)
    date_col = _date_col(market_data)
    market_dates = pd.to_datetime(market_data[date_col], errors="coerce")
    market = market_data.copy()
    market[date_col] = market_dates
    market = market[market[date_col].notna()].copy()
    if market.empty:
        return pd.DataFrame(columns=cols)
    train_start_cfg = pd.to_datetime(split_cfg["walk_forward_initial_train_start"])
    train_end_cfg = pd.to_datetime(split_cfg["walk_forward_initial_train_end"])
    test_freq = str(split_cfg.get("walk_forward_test_frequency", "yearly")).lower()
    if test_freq != "yearly":
        raise ValueError("Only yearly walk-forward frequency is supported.")
    expanding = bool(split_cfg.get("expanding_window", True))

    initial_train_rows = market[(market[date_col] >= train_start_cfg) & (market[date_col] <= train_end_cfg)]
    if initial_train_rows.empty:
        return pd.DataFrame([{
            "split_id": "wf_000",
            "split_type": "full_walk_forward",
            "regime_name": "full_period_walk_forward",
            "regime_label": "Expanding-window full-period walk-forward",
            "train_start": train_start_cfg,
            "train_end": train_end_cfg,
            "test_start": pd.NaT,
            "test_end": pd.NaT,
            "expanding_window": expanding,
            "split_status": "insufficient_initial_training_window",
            "n_train_market_rows": 0,
            "n_test_market_rows": 0,
        }], columns=cols)

    years = sorted(market[date_col].dt.year.unique().tolist())
    first_test_year = int(train_end_cfg.year + 1)
    last_test_year = int(max(years))
    rows = []
    split_idx = 1
    train_span_years = int(train_end_cfg.year - train_start_cfg.year + 1)
    for test_year in range(first_test_year, last_test_year + 1):
        if expanding:
            train_start = train_start_cfg
        else:
            rolling_start_year = test_year - train_span_years
            train_start = pd.Timestamp(f"{rolling_start_year}-01-01")
        train_end = pd.Timestamp(f"{test_year - 1}-12-31")
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")
        n_train = int(((market[date_col] >= train_start) & (market[date_col] <= train_end)).sum())
        n_test = int(((market[date_col] >= test_start) & (market[date_col] <= test_end)).sum())
        split_status = "ok" if n_train > 0 and n_test > 0 else "insufficient_market_rows"
        rows.append({
            "split_id": f"wf_{split_idx:03d}",
            "split_type": "full_walk_forward",
            "regime_name": "full_period_walk_forward",
            "regime_label": "Expanding-window full-period walk-forward",
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "expanding_window": expanding,
            "split_status": split_status,
            "n_train_market_rows": n_train,
            "n_test_market_rows": n_test,
        })
        split_idx += 1
    if not rows:
        rows = [{
            "split_id": "wf_000",
            "split_type": "full_walk_forward",
            "regime_name": "full_period_walk_forward",
            "regime_label": "Expanding-window full-period walk-forward",
            "train_start": train_start_cfg,
            "train_end": train_end_cfg,
            "test_start": pd.NaT,
            "test_end": pd.NaT,
            "expanding_window": expanding,
            "split_status": "insufficient_initial_training_window",
            "n_train_market_rows": int(len(initial_train_rows)),
            "n_test_market_rows": 0,
        }]
    return pd.DataFrame(rows, columns=cols)


def build_regime_train_test_splits(market_data: pd.DataFrame, config: dict) -> pd.DataFrame:
    spec = config.get("regime_backtest", {})
    if not spec.get("enabled", False):
        return pd.DataFrame(columns=[
            "split_id", "split_type", "regime_name", "regime_label",
            "train_start", "train_end", "test_start", "test_end",
            "regime_start", "regime_end", "expanding_window", "split_status",
            "n_train_market_rows", "n_test_market_rows",
        ])
    regimes = spec.get("regimes", {})
    date_col = _date_col(market_data) if not market_data.empty else None
    market_dates = pd.to_datetime(market_data[date_col], errors="coerce") if date_col else pd.Series(dtype="datetime64[ns]")
    rows = []
    for idx, (name, reg) in enumerate(regimes.items(), start=1):
        train_start = _to_ts(reg["train_start"])
        train_end = _to_ts(reg["train_end"])
        test_start = _to_ts(reg["test_start"])
        test_end = _to_ts(reg["test_end"])
        if date_col:
            n_train = int(((market_dates >= train_start) & (market_dates <= train_end)).sum())
            n_test = int(((market_dates >= test_start) & (market_dates <= test_end)).sum())
        else:
            n_train = 0
            n_test = 0
        rows.append({
            "split_id": f"reg_{idx:03d}",
            "split_type": "regime_train_test",
            "regime_name": name,
            "regime_label": reg.get("label", name),
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "regime_start": _to_ts(reg["regime_start"]),
            "regime_end": _to_ts(reg["regime_end"]),
            "expanding_window": False,
            "split_status": "ok" if n_train > 0 and n_test > 0 else "insufficient_market_rows",
            "n_train_market_rows": n_train,
            "n_test_market_rows": n_test,
        })
    return pd.DataFrame(rows)


def assign_split_id(df: pd.DataFrame, split_windows: pd.DataFrame, date_column: str = "valuation_date") -> pd.DataFrame:
    out = df.copy()
    out["split_id"] = None
    out["split_type"] = None
    out["regime_name"] = None
    out["regime_label"] = None
    if out.empty or split_windows.empty:
        return out
    date_series = pd.to_datetime(out[date_column])
    for row in split_windows.to_dict("records"):
        mask = (date_series >= pd.to_datetime(row["test_start"])) & (date_series <= pd.to_datetime(row["test_end"]))
        out.loc[mask, "split_id"] = row["split_id"]
        out.loc[mask, "split_type"] = row["split_type"]
        out.loc[mask, "regime_name"] = row["regime_name"]
        out.loc[mask, "regime_label"] = row["regime_label"]
    return out


def validate_chronological_splits(split_windows: pd.DataFrame) -> tuple[bool, list[dict]]:
    violations = []
    for row in split_windows.to_dict("records"):
        train_start = pd.to_datetime(row["train_start"])
        train_end = pd.to_datetime(row["train_end"])
        test_start = pd.to_datetime(row["test_start"])
        test_end = pd.to_datetime(row["test_end"])
        if not (train_start <= train_end < test_start <= test_end):
            violations.append(row)
    return len(violations) == 0, violations


def validate_no_train_test_overlap(split_windows: pd.DataFrame) -> tuple[bool, list[dict]]:
    violations = []
    for row in split_windows.to_dict("records"):
        train_start = pd.to_datetime(row["train_start"])
        train_end = pd.to_datetime(row["train_end"])
        test_start = pd.to_datetime(row["test_start"])
        test_end = pd.to_datetime(row["test_end"])
        if max(train_start, test_start) <= min(train_end, test_end):
            violations.append(row)
    return len(violations) == 0, violations
