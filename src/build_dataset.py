"""Turn raw NOAA downloads into model-ready tables.

Outputs (in data/processed/):
  seasons.csv   one row per season: what actually happened (the targets)
  features.csv  one row per (season, checkpoint): what was knowable on that date (the inputs)
  daily.csv     cleaned daily series used for each season (for charts)

Usage:  python -m src.build_dataset
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

MM_PER_IN = 25.4


# --------------------------------------------------------------------------- loading
def _qc_ok(attr: pd.Series) -> pd.Series:
    """GHCN attribute strings look like 'mflag,qflag,sflag,time'. A non-blank qflag = failed quality check."""
    q = attr.fillna("").astype(str).str.split(",").str[1].fillna("").str.strip()
    return q == ""


def load_station(path: Path) -> pd.DataFrame:
    """Load one GHCN-Daily 'access' CSV and convert to inches / Fahrenheit.

    NOAA stores SNOW and SNWD in millimetres and TMAX/TMIN in tenths of a degree Celsius.
    """
    raw = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(index=pd.to_datetime(raw["DATE"]))
    out.index.name = "date"

    def element(name: str) -> pd.Series:
        if name not in raw.columns:
            return pd.Series(np.nan, index=out.index)
        vals = pd.to_numeric(raw[name], errors="coerce")
        attr_col = f"{name}_ATTRIBUTES"
        if attr_col in raw.columns:
            vals = vals.where(_qc_ok(raw[attr_col]))
        return pd.Series(vals.to_numpy(), index=out.index)

    out["snow_in"] = element("SNOW") / MM_PER_IN
    out["depth_in"] = element("SNWD") / MM_PER_IN
    out["tmax_f"] = element("TMAX") / 10 * 9 / 5 + 32
    out["tmin_f"] = element("TMIN") / 10 * 9 / 5 + 32
    out = out[out.index.notna() & ~out.index.duplicated()].sort_index()
    return out


def check_units(daily: pd.DataFrame, station_id: str) -> None:
    """Warn loudly if converted values look implausible (catches unit mix-ups)."""
    winter = daily[daily.index.month.isin([12, 1, 2])]
    if winter["tmax_f"].notna().sum() > 100:
        m = winter["tmax_f"].median()
        if not 15 < m < 60:
            print(f"  WARNING {station_id}: median winter high is {m:.0f}F - check temperature units")
    yearly = daily["snow_in"].groupby(daily.index.year).sum(min_count=200).dropna()
    if len(yearly) > 5 and not 5 < yearly.median() < 150:
        print(f"  WARNING {station_id}: median annual snowfall is {yearly.median():.0f} in - check snow units")


# --------------------------------------------------------------------------- seasons
def season_bounds(season: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    (sm, sd), (em, ed) = config.SEASON_START, config.SEASON_END
    return pd.Timestamp(season, sm, sd), pd.Timestamp(season + 1, em, ed)


def checkpoint_date(season: int, cp: tuple) -> pd.Timestamp:
    month, day = cp
    return pd.Timestamp(season if month >= 7 else season + 1, month, day)


def _count_where(s: pd.Series, cond: pd.Series, n_days: int) -> float:
    """Count days meeting cond, or NaN if the series is too gappy to trust."""
    if n_days == 0:
        return 0.0
    if s.notna().sum() / n_days < config.MIN_COMPLETENESS:
        return np.nan
    return float(cond.sum())


def season_summary(daily: pd.DataFrame, season: int) -> dict:
    start, end = season_bounds(season)
    w = daily.loc[start:end]
    n_days = (end - start).days + 1
    completeness = w["snow_in"].notna().sum() / n_days
    return {
        "season": season,
        "completeness": completeness,
        "snowfall_in": w["snow_in"].sum() if completeness >= config.MIN_COMPLETENESS else np.nan,
        "depth_days": _count_where(w["depth_in"], w["depth_in"] >= config.DEPTH_DAY_THRESHOLD_IN, n_days),
        "snowmaking_nights": _count_where(w["tmin_f"], w["tmin_f"] <= config.SNOWMAKING_TMIN_F, n_days),
    }


def pick_sources(summaries: dict[str, pd.DataFrame], stations: pd.DataFrame) -> pd.DataFrame:
    """Choose which station supplies each season.

    The primary station is the one with the most complete seasons. When it has a gap, the nearest
    neighbour with a complete record fills in, rescaled by the median primary/neighbour snowfall ratio
    over seasons both stations observed (stations at different elevations get different snow).
    """
    order = stations.sort_values("distance_km")["station_id"].tolist()
    n_complete = {sid: summaries[sid]["snowfall_in"].notna().sum() for sid in order}
    primary = max(order, key=lambda s: (n_complete[s], -order.index(s)))

    p = summaries[primary].set_index("season")["snowfall_in"]
    scale = {primary: 1.0}
    for sid in order:
        if sid == primary:
            continue
        n = summaries[sid].set_index("season")["snowfall_in"]
        both = pd.concat([p, n], axis=1, keys=["p", "n"]).dropna()
        both = both[both["n"] > 0]
        if len(both) >= 5:
            scale[sid] = float((both["p"] / both["n"]).median())

    all_seasons = sorted(set().union(*[set(df["season"]) for df in summaries.values()]))
    rows = []
    for season in all_seasons:
        source = None
        for sid in [primary] + [s for s in order if s != primary and s in scale]:
            row = summaries[sid].set_index("season")
            if season in row.index and not np.isnan(row.loc[season, "snowfall_in"]):
                source = sid
                break
        if source is None:  # nothing complete: use primary anyway (e.g. the season in progress)
            source = primary
        rows.append({"season": season, "source_station": source, "scale": scale.get(source, 1.0),
                     "filled_from_neighbor": source != primary})
    print(f"  primary station: {primary}   neighbour scale factors: "
          + ", ".join(f"{k}={v:.2f}" for k, v in scale.items() if k != primary))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- climate indices
ONI_CENTER_MONTH = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
                    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def _first_of_month_after(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year + (month == 12), month % 12 + 1, 1)


def load_oni(path: Path) -> pd.DataFrame:
    """ONI = 3-month running mean of tropical Pacific SST anomaly (El Nino > +0.5, La Nina < -0.5)."""
    df = pd.read_csv(path, sep=r"\s+")
    rows = []
    for seas, yr, anom in zip(df["SEAS"], df["YR"], df["ANOM"]):
        c = ONI_CENTER_MONTH[seas]
        end_year, end_month = (yr + 1, 1) if c == 12 else (yr, c + 1)
        rows.append({"available_from": _first_of_month_after(end_year, end_month), "value": float(anom)})
    return pd.DataFrame(rows).sort_values("available_from").reset_index(drop=True)


def load_monthly_index(path: Path) -> pd.DataFrame:
    """CPC monthly index files: 'year month value' per line."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["year", "month", "value"])
    df = df[pd.to_numeric(df["value"], errors="coerce").between(-20, 20)]
    df["available_from"] = [_first_of_month_after(int(y), int(m)) for y, m in zip(df["year"], df["month"])]
    return df[["available_from", "value"]].astype({"value": float}).sort_values("available_from").reset_index(drop=True)


def index_as_of(idx: pd.DataFrame, when: pd.Timestamp, n_recent: int = 1) -> float:
    known = idx[idx["available_from"] <= when]
    return float(known["value"].tail(n_recent).mean()) if len(known) else np.nan


# --------------------------------------------------------------------------- features
def checkpoint_features(daily: pd.DataFrame, season: int, cp: tuple, scale: float,
                        tavg_clim: pd.Series, indices: dict) -> dict:
    """Everything knowable on the checkpoint date. Only data strictly before cp_date is used."""
    cp_date = checkpoint_date(season, cp)
    since = pd.Timestamp(season, 10, 1)  # count early-season snow from Oct 1
    w = daily.loc[since: cp_date - pd.Timedelta(days=1)]
    n_days = (cp_date - since).days

    snow_ok = n_days > 0 and w["snow_in"].notna().sum() / n_days >= config.MIN_COMPLETENESS
    tavg = (w["tmax_f"] + w["tmin_f"]) / 2
    doy = w.index.dayofyear
    tavg_anom = (tavg - tavg_clim.reindex(doy).to_numpy()).mean() if tavg.notna().sum() > 0.7 * n_days else np.nan

    return {
        "season": season,
        "checkpoint": config.checkpoint_label(cp),
        "checkpoint_date": cp_date.date(),
        "snow_to_date_in": w["snow_in"].sum() * scale if snow_ok else np.nan,
        "depth_days_to_date": _count_where(w["depth_in"], w["depth_in"] >= config.DEPTH_DAY_THRESHOLD_IN, n_days),
        "snowmaking_nights_to_date": _count_where(w["tmin_f"], w["tmin_f"] <= config.SNOWMAKING_TMIN_F, n_days),
        "temp_anom_to_date_f": tavg_anom,
        "oni": index_as_of(indices["oni"], cp_date, 1),
        "nao_recent": index_as_of(indices["nao"], cp_date, 2),
        "ao_recent": index_as_of(indices["ao"], cp_date, 2),
    }


def main(raw_dir: Path = config.RAW_DIR, out_dir: Path = config.PROCESSED_DIR, today: dt.date | None = None) -> None:
    today = pd.Timestamp(today or dt.date.today())
    out_dir.mkdir(parents=True, exist_ok=True)
    stations = pd.read_csv(raw_dir / "stations.csv", dtype={"station_id": str})

    print("Loading stations")
    dailies, summaries = {}, {}
    for sid in stations["station_id"]:
        path = raw_dir / "ghcn" / f"{sid}.csv"
        if not path.exists():
            print(f"  missing {path.name}, skipping")
            continue
        d = load_station(path)
        check_units(d, sid)
        dailies[sid] = d
    if not dailies:
        raise RuntimeError("No station data found - run python -m src.fetch_data first")

    # Same season range for every station so they can be compared / substituted.
    first = min(d.index.min().year for d in dailies.values())
    last = today.year if today.month >= 7 else today.year - 1
    for sid, d in dailies.items():
        summaries[sid] = pd.DataFrame([season_summary(d, s) for s in range(first, last + 1)])
        print(f"  {sid}: {summaries[sid]['snowfall_in'].notna().sum()} complete seasons")
    stations = stations[stations["station_id"].isin(dailies)]

    sources = pick_sources(summaries, stations)
    indices = {"oni": load_oni(raw_dir / "oni.txt"),
               "nao": load_monthly_index(raw_dir / "nao.txt"),
               "ao": load_monthly_index(raw_dir / "ao.txt")}

    clim = {}
    for sid, d in dailies.items():
        tavg = (d["tmax_f"] + d["tmin_f"]) / 2
        clim[sid] = tavg.groupby(d.index.dayofyear).mean().rolling(15, center=True, min_periods=1).mean()

    season_rows, feature_rows, daily_rows = [], [], []
    for _, src in sources.iterrows():
        season, sid, scale = int(src["season"]), src["source_station"], float(src["scale"])
        d = dailies[sid]
        summ = summaries[sid].set_index("season").loc[season].to_dict()
        start, end = season_bounds(season)
        in_progress = start - pd.Timedelta(days=31) <= today <= end
        if summ["completeness"] == 0 and not in_progress:
            continue  # no station was reporting yet
        season_rows.append({
            "season": season, "label": f"{season}-{str(season + 1)[-2:]}",
            "source_station": sid, "filled_from_neighbor": bool(src["filled_from_neighbor"]),
            "in_progress": in_progress,
            "snowfall_in": np.nan if in_progress else summ["snowfall_in"] * scale,
            "depth_days": np.nan if in_progress else summ["depth_days"],
            "snowmaking_nights": np.nan if in_progress else summ["snowmaking_nights"],
        })
        for cp in config.CHECKPOINTS:
            if checkpoint_date(season, cp) <= today + pd.Timedelta(days=1):
                feature_rows.append(checkpoint_features(d, season, cp, scale, clim[sid], indices))
        w = d.loc[pd.Timestamp(season, 10, 1):end, ["snow_in", "depth_in"]].copy()
        w["snow_in"] *= scale
        w["season"] = season
        daily_rows.append(w)

    seasons = pd.DataFrame(season_rows)
    features = pd.DataFrame(feature_rows)
    seasons.to_csv(out_dir / "seasons.csv", index=False)
    features.to_csv(out_dir / "features.csv", index=False)
    pd.concat(daily_rows).reset_index().to_csv(out_dir / "daily.csv", index=False)

    done = seasons.dropna(subset=[config.TARGET])
    print(f"\n{len(done)} completed seasons with a usable target ({done['season'].min()}-{done['season'].max()}),"
          f" {int(seasons['filled_from_neighbor'].sum())} filled from neighbour stations")
    print(f"Median {config.TARGET}: {done[config.TARGET].median():.1f}")
    print("Next: python -m src.train")


if __name__ == "__main__":
    main()
