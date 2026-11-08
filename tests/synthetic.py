"""Generate fake NOAA-format files so the pipeline can be tested without internet access.

The fake snowfall has a built-in, known signal (a weak climate-index effect plus a strong
"snowy early season -> snowy full season" effect), so tests can check the model finds it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEAS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]


def make_raw(raw_dir: Path, start_year: int = 1965, end: str = "2026-11-06", seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "ghcn").mkdir(exist_ok=True)
    years = range(start_year, int(end[:4]) + 1)

    # Monthly climate indices
    months = pd.period_range(f"{start_year}-01", end[:7], freq="M")
    nao = pd.Series(rng.normal(0, 1, len(months)), index=months)
    ao = pd.Series(0.6 * nao.to_numpy() + rng.normal(0, 0.8, len(months)), index=months)
    oni_m = pd.Series(np.cumsum(rng.normal(0, 0.25, len(months))) * 0.5, index=months).clip(-2.5, 2.5)
    with open(raw_dir / "nao.txt", "w") as f:
        f.writelines(f"{p.year:5d}{p.month:5d}{v:8.2f}\n" for p, v in nao.items())
    with open(raw_dir / "ao.txt", "w") as f:
        f.writelines(f"{p.year:5d}{p.month:5d}{v:8.3f}\n" for p, v in ao.items())
    with open(raw_dir / "oni.txt", "w") as f:
        f.write("SEAS  YR   TOTAL   ANOM\n")
        for p, v in oni_m.items():
            f.write(f"{SEAS[p.month - 1]:>4} {p.year:4d} {27 + v:7.2f} {v:7.2f}\n")

    # Season "snowiness": weakly tied to fall NAO, so early snow predicts the rest of the season
    snowiness = {}
    for y in years:
        key = pd.Period(f"{y}-10", freq="M")
        signal = -0.3 * nao.get(key, 0.0)
        snowiness[y] = float(np.exp(0.35 * rng.normal() + 0.15 * signal))

    dates = pd.date_range(f"{start_year}-01-01", end, freq="D")
    doy = dates.dayofyear.to_numpy()
    season_year = np.where(dates.month >= 7, dates.year, dates.year - 1)
    base_temp = 50 - 25 * np.cos(2 * np.pi * (doy - 15) / 365)

    stations = []
    for k, (sid, dist, mult) in enumerate([("USC00999001", 5.0, 1.0), ("USC00999002", 12.0, 0.8),
                                           ("USC00999003", 25.0, 1.2)]):
        sn = np.array([snowiness.get(y, 1.0) for y in season_year])
        cold = np.clip((40 - base_temp) / 15, 0, None)
        snow_in = np.where(rng.random(len(dates)) < 0.12 * cold * sn, rng.gamma(1.5, 2.5, len(dates)) * sn * mult, 0.0)
        depth = np.zeros(len(dates))
        for i in range(1, len(dates)):
            depth[i] = max(0.0, depth[i - 1] * 0.93 + snow_in[i] - 0.3 * max(0, base_temp[i] - 32) / 5)
        tmax_f = base_temp + 8 + rng.normal(0, 6, len(dates)) - 3 * (sn - 1)
        tmin_f = tmax_f - 18

        df = pd.DataFrame({
            "STATION": sid, "DATE": dates.strftime("%Y-%m-%d"),
            "SNOW": np.round(snow_in * 25.4), "SNOW_ATTRIBUTES": ",,7,0700",
            "SNWD": np.round(depth * 25.4), "SNWD_ATTRIBUTES": ",,7,0700",
            "TMAX": np.round((tmax_f - 32) * 5 / 9 * 10), "TMAX_ATTRIBUTES": ",,7,0700",
            "TMIN": np.round((tmin_f - 32) * 5 / 9 * 10), "TMIN_ATTRIBUTES": ",,7,0700",
        })
        # Gaps: station 1 missing a few whole seasons (so neighbours must fill in), random missing days
        if k == 0:
            gap = (dates >= "1988-10-01") & (dates <= "1991-05-01")
            df.loc[gap, ["SNOW", "SNWD"]] = np.nan
        else:  # neighbours have shorter records, so station 1 stays the primary
            df = df[dates >= "1980-01-01"]
        df.loc[rng.random(len(df)) < 0.03, "SNOW"] = np.nan
        df.iloc[5, df.columns.get_loc("SNOW_ATTRIBUTES")] = ",X,7,0700"  # one failed-QC value
        df.to_csv(raw_dir / "ghcn" / f"{sid}.csv", index=False)
        stations.append({"station_id": sid, "lat": 41.18, "lon": -74.5, "elev_m": 300, "state": "NJ",
                         "name": f"FAKE STATION {k + 1}", "first_year": start_year, "last_year": 2026,
                         "snow_years": 60, "distance_km": dist})
    pd.DataFrame(stations).to_csv(raw_dir / "stations.csv", index=False)
