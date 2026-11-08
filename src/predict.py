"""Forecast the season in progress using the latest data.

Usage:  python -m src.predict
(Run python -m src.fetch_data && python -m src.build_dataset first to pick up the newest observations.)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src import config


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
            "checkpoint": row["checkpoint"],
            "p_above_median": float(m["classifier"].predict_proba(X)[0, 1]),
            "expected_total": float(m["regressor"].predict(X)[0]),
            "median_season": m["threshold"],
            "snow_to_date_in": row.get("snow_to_date_in"),
        })
    return pd.DataFrame(out)


def main() -> None:
    fc = current_forecast()
    if fc.empty:
        print("No checkpoint reached yet this season - the first forecast is issued on",
              config.checkpoint_label(config.CHECKPOINTS[0]))
        return
    latest = fc.iloc[-1]
    print(f"\n{config.RESORT_NAME} {latest['season']} forecast (as of {latest['checkpoint']})")
    print(f"  Chance of an above-median season: {latest['p_above_median']:.0%}")
    print(f"  Expected {bundle_target()}: {latest['expected_total']:.0f}   (median season: {latest['median_season']:.0f})")
    print("\nAll checkpoints so far:")
    print(fc.round(2).to_string(index=False))
    config.REPORTS_DIR.mkdir(exist_ok=True)
    fc.to_csv(config.REPORTS_DIR / "current_forecast.csv", index=False)


def bundle_target() -> str:
    return {"snowfall_in": "season snowfall (in)", "depth_days": "snow-on-ground days",
            "snowmaking_nights": "snowmaking nights"}.get(config.TARGET, config.TARGET)


if __name__ == "__main__":
    main()
