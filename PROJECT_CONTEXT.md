# Project context: Mountain Creek Season Forecaster

Resume project: a probabilistic forecaster for whether the snowboarding season at Mountain Creek
(Vernon, NJ) will be "good" (above the historical median). Local repo: `~/Coding/mountain-creek-snow-forecast`.
Run from VS Code's terminal inside a `.venv`, always using `python -m src.<module>` (never the ▶️ Run button).

## Status
- Pipeline built and tested end to end on synthetic data (7 pytest tests pass).
- Not yet run on real NOAA data. Next step: run fetch_data → build_dataset → train, then fill in README results.

## Design decisions
- Target (config.TARGET): `snowfall_in` (Nov 1–Apr 30 total). Alternatives: `depth_days`, `snowmaking_nights`
  (nights ≤25°F; relevant because Mountain Creek relies heavily on snowmaking).
- Binary target: above/below median, with the median recomputed inside each CV fold (no leakage).
- Checkpoints: Nov 1, Dec 1, Dec 15, Jan 1, Jan 15, Feb 1. Features only use data before each date.
- Features: snow since Oct 1, snow-on-ground days, snowmaking nights, temp anomaly, ONI, NAO and AO (mean of last 2 months).
- Models: logistic regression (C=0.3) for P(above median), ridge (alpha=5) for expected total.
  Variants: `climate_indices_only` vs `full`.
- Evaluation: leave-one-season-out CV, Brier Skill Score vs climatology with 90% bootstrap CI, plus MAE.
- Stations: auto-selected NOAA GHCN-Daily stations within 45 km with ≥25 yrs of SNOW. The primary station is the
  one with the most complete seasons; gaps are filled from neighbours scaled by the median snowfall ratio.
- Units: GHCN SNOW/SNWD are in mm and TMAX/TMIN in tenths of °C, converted to inches and °F. check_units() warns
  if the converted values look implausible.

## Files
- src/config.py: resort location, season definition, target, checkpoints, URLs
- src/fetch_data.py: station discovery + downloads (NOAA NCEI + CPC)
- src/build_dataset.py: outputs data/processed/seasons.csv, features.csv, daily.csv
- src/train.py: outputs reports/cv_results.csv, cv_predictions.csv, skill_by_checkpoint.png, models/models.joblib
- src/predict.py: current-season forecast (reports/current_forecast.csv)
- app.py: Streamlit dashboard
- tests/: synthetic NOAA-format data generator + end-to-end tests

## Ideas for later
- Try snowmaking_nights as the target; compare to snowfall.
- Deploy the dashboard (Streamlit Community Cloud) and update it weekly during the season.
- Check the live 2026-27 forecast against reality in April.
