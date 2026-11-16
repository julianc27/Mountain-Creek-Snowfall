"""Forecast the season in progress using the latest data.

Usage:  python -m src.predict
(Run python -m src.fetch_data && python -m src.build_dataset first to pick up the newest observations.)

Two outputs:
  reports/current_forecast.csv  every checkpoint so far, recomputed from today's data (overwritten each run)
  reports/forecast_log.csv      the official live record: each checkpoint is logged the first time it is
                                forecast and never changed afterwards, so it can be scored honestly in April
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src.train import FORECAST_LOG

LOG_KEY = ["season_start", "checkpoint", "model"]


def current_forecast(processed_dir: Path = config.PROCESSED_DIR, models_dir: Path = config.MODELS_DIR,
                     model_name: str = "full") -> pd.DataFrame:
    bundle = joblib.load(models_dir / "models.joblib")
    seasons = pd.read_csv(processed_dir / "seasons.csv")
    features = pd.read_csv(processed_dir / "features.csv")

    live = seasons[seasons["in_progress"]]
    if live.empty:
        raise RuntimeError("No season in progress (the off-season runs May-September).")
    season = int(live["season"].iloc[0])
    rows = features[features["season"] == season]

    out = []
    for _, row in rows.iterrows():
        m = bundle["models"].get(row["checkpoint"], {}).get(model_name)
        if m is None:
            continue
        X = row[m["features"]].to_frame().T.astype(float)
        out.append({
            "season": f"{season}-{str(season + 1)[-2:]}",
            "season_start": season,
            "checkpoint": row["checkpoint"],
            "model": model_name,
            "model_version": bundle.get("version", "unversioned"),
            "p_above_median": float(m["classifier"].predict_proba(X)[0, 1]),
            "expected_total": float(m["regressor"].predict(X)[0]),
            "median_season": m["threshold"],
            "snow_to_date_in": row.get("snow_to_date_in"),
        })
    return pd.DataFrame(out)


def log_forecast(fc: pd.DataFrame, log_path: Path, issued_at: dt.datetime | None = None) -> pd.DataFrame:
    """Append checkpoints that aren't in the log yet. Rows already logged are never modified.

    Returns the newly added rows (empty if nothing new).
    """
    if fc.empty:
        return fc
    issued_at = issued_at or dt.datetime.now(dt.timezone.utc)
    old = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame(columns=["issued_at", *fc.columns])

    logged = set(map(tuple, old[LOG_KEY].astype(str).to_numpy())) if len(old) else set()
    is_new = [tuple(map(str, k)) not in logged for k in fc[LOG_KEY].to_numpy()]
    new = fc[is_new].copy()
    if new.empty:
        return new
    new.insert(0, "issued_at", issued_at.isoformat(timespec="seconds"))
    if log_path.exists():
        new = new.reindex(columns=old.columns)  # keep the file's column order if the code has changed since

    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Append rather than rewrite, so earlier lines in the file are physically untouched.
    new.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    return new


def main() -> None:
    fc = current_forecast()
    if fc.empty:
        print("No checkpoint reached yet this season - the first forecast is issued on",
              config.checkpoint_label(config.CHECKPOINTS[0]))
        return
    latest = fc.iloc[-1]
    print(f"\n{config.RESORT_NAME} {latest['season']} forecast (as of {latest['checkpoint']}, "
          f"model {latest['model_version']})")
    print(f"  Chance of an above-median season: {latest['p_above_median']:.0%}")
    print(f"  Expected {bundle_target()}: {latest['expected_total']:.0f}   (median season: {latest['median_season']:.0f})")
    print("\nAll checkpoints so far (recomputed with today's data):")
    print(fc.drop(columns=["season_start", "model"]).round(2).to_string(index=False))

    config.REPORTS_DIR.mkdir(exist_ok=True)
    fc.to_csv(config.REPORTS_DIR / "current_forecast.csv", index=False)

    new = log_forecast(fc, config.REPORTS_DIR / FORECAST_LOG)
    if new.empty:
        print(f"\nNo new checkpoint since the last run - {FORECAST_LOG} unchanged.")
    else:
        print(f"\nLogged to {FORECAST_LOG}: " + ", ".join(new["checkpoint"]) + ". Commit it so the timestamp is public.")


def bundle_target() -> str:
    return {"snowfall_in": "season snowfall (in)", "depth_days": "snow-on-ground days",
            "snowmaking_nights": "snowmaking nights"}.get(config.TARGET, config.TARGET)


if __name__ == "__main__":
    main()
