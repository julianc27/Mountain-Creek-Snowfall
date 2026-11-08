# 🏂 Mountain Creek Season Forecaster

**Will this winter be a good one at Mountain Creek (Vernon, NJ)?** This project gives a probabilistic answer
built from 50+ years of NOAA weather-station records and large-scale climate patterns. It forecasts before
the season starts and updates as the winter goes on.

> Status: pipeline complete and tested; results below get filled in after the first real-data run.

## The question

A season is "good" if it finishes **above the historical median** for the chosen target (set in `src/config.py`):

| Target | What it measures |
|---|---|
| `snowfall_in` (default) | Total natural snowfall, Nov 1 – Apr 30 |
| `depth_days` | Days with ≥1" of snow on the ground |
| `snowmaking_nights` | Nights ≤25°F. For an East Coast resort, cold for snowmaking matters as much as natural snow |

Forecasts are issued at six **checkpoints** (Nov 1, Dec 1, Dec 15, Jan 1, Jan 15, Feb 1), and each one uses
only information that was actually available on that date.

## Data (all free, public)

- **NOAA GHCN-Daily**: daily snowfall, snow depth and temperature. `fetch_data.py` automatically finds the
  nearest stations to the resort with ≥25 years of snowfall records.
- **ENSO / ONI** (El Niño–La Niña), **NAO** (North Atlantic Oscillation) and **AO** (Arctic Oscillation)
  from NOAA's Climate Prediction Center.

## Method

1. **Station blending.** The station with the most complete record is primary. Seasons it missed are filled from
   the nearest neighbour, rescaled by the median ratio between the two stations (they sit at different elevations).
2. **Features at each checkpoint:** snow since Oct 1, snow-on-ground days, snowmaking nights, temperature
   anomaly, latest ONI, recent NAO/AO.
3. **Models:** regularized logistic regression (probability of an above-median season) and ridge regression
   (expected total). There are two variants: *climate indices only* and *full* (indices plus season-to-date
   observations).
4. **Evaluation:** leave-one-season-out cross-validation. The median threshold is recomputed inside each
   fold to avoid leakage. Skill is reported as the **Brier Skill Score vs. climatology** (always guessing the
   base rate), with 90% bootstrap confidence intervals.

## Results

_Run the pipeline, then paste in `reports/skill_by_checkpoint.png` and the key numbers from `reports/cv_results.csv`._

![skill](reports/skill_by_checkpoint.png)

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.fetch_data      # download NOAA data (a few minutes the first time)
python -m src.build_dataset   # clean + build season targets and checkpoint features
python -m src.train           # cross-validate, save models + reports
python -m src.predict         # forecast the season in progress
streamlit run app.py          # dashboard

python -m pytest -q           # end-to-end tests on synthetic data (no internet needed)
```

During the season, re-run `fetch_data → build_dataset → predict` to update the forecast with new observations.

## Project structure

```
src/config.py         resort location, season definition, target, data URLs
src/fetch_data.py     station discovery + downloads
src/build_dataset.py  unit conversion, QC, station blending, targets, features
src/train.py          cross-validation, skill scores, final models
src/predict.py        current-season forecast
app.py                Streamlit dashboard
tests/                synthetic NOAA-format data + end-to-end tests
```

## Limitations

- Only a few dozen seasons are available, so any skill estimate has wide error bars. That's why the
  confidence intervals are reported.
- Natural snowfall at a valley station is not the same as conditions on the slopes, and Mountain Creek relies
  heavily on snowmaking.
- Climate indices have only a modest, region-dependent effect on Northeast winters.
