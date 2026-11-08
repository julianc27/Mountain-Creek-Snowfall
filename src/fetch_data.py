"""Download raw data from NOAA.

1. Finds the NOAA GHCN-Daily weather stations nearest Mountain Creek that have a long snowfall record.
2. Downloads each station's full daily history (snowfall, snow depth, temperature, precipitation).
3. Downloads climate indices: ENSO (ONI), North Atlantic Oscillation (NAO), Arctic Oscillation (AO).

Usage:  python -m src.fetch_data
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import time
import urllib.request
from pathlib import Path

import pandas as pd

from src import config

USER_AGENT = "mountain-creek-snow-forecast (student research project)"


def download(url: str, dest: Path, force: bool = False, retries: int = 3) -> Path:
    """Download url to dest, skipping if it already exists (unless force)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"  cached  {dest.relative_to(config.ROOT)}")
        return dest
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            dest.write_bytes(data)
            print(f"  fetched {dest.relative_to(config.ROOT)} ({len(data) / 1e6:.1f} MB)")
            return dest
        except Exception as exc:  # network hiccups are common with NOAA
            print(f"  attempt {attempt} failed for {url}: {exc}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not download {url}")


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def read_station_list(path: Path) -> pd.DataFrame:
    # Fixed-width format documented in NOAA's GHCN-Daily readme.
    colspecs = [(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71)]
    names = ["station_id", "lat", "lon", "elev_m", "state", "name"]
    return pd.read_fwf(path, colspecs=colspecs, names=names, dtype={"station_id": str, "state": str})


def read_inventory(path: Path) -> pd.DataFrame:
    colspecs = [(0, 11), (31, 35), (36, 40), (41, 45)]
    names = ["station_id", "element", "first_year", "last_year"]
    return pd.read_fwf(path, colspecs=colspecs, names=names, dtype={"station_id": str})


def select_stations(stations: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    """Pick the nearest stations that have a long, still-active snowfall record."""
    this_year = dt.date.today().year
    snow = inventory[inventory["element"] == "SNOW"].copy()
    snow["snow_years"] = snow["last_year"] - snow["first_year"] + 1
    snow = snow[(snow["snow_years"] >= config.MIN_SNOW_YEARS)
                & (snow["last_year"] >= this_year - config.MIN_LAST_YEAR_LAG)]

    cand = stations.merge(snow[["station_id", "first_year", "last_year", "snow_years"]], on="station_id")
    cand["distance_km"] = [
        haversine_km(config.RESORT_LAT, config.RESORT_LON, la, lo) for la, lo in zip(cand["lat"], cand["lon"])
    ]
    cand = cand[cand["distance_km"] <= config.SEARCH_RADIUS_KM]
    cand = cand.sort_values("distance_km").head(config.MAX_STATIONS).reset_index(drop=True)
    if cand.empty:
        raise RuntimeError("No qualifying stations found - try increasing SEARCH_RADIUS_KM in config.py")
    return cand


def main(force: bool = False) -> None:
    raw = config.RAW_DIR
    print("1/3  Station catalogue")
    st_path = download(config.GHCN_STATIONS_URL, raw / "ghcnd-stations.txt", force)
    inv_path = download(config.GHCN_INVENTORY_URL, raw / "ghcnd-inventory.txt", force)
    chosen = select_stations(read_station_list(st_path), read_inventory(inv_path))
    chosen.to_csv(raw / "stations.csv", index=False)
    print("\nStations selected near", config.RESORT_NAME)
    print(chosen[["station_id", "name", "distance_km", "elev_m", "first_year", "last_year"]]
          .round(1).to_string(index=False))

    print("\n2/3  Daily station histories (re-downloaded every run so the current season stays fresh)")
    for sid in chosen["station_id"]:
        download(config.GHCN_STATION_CSV_URL.format(station_id=sid), raw / "ghcn" / f"{sid}.csv", force=True)

    print("\n3/3  Climate indices")
    download(config.ONI_URL, raw / "oni.txt", force=True)
    download(config.NAO_URL, raw / "nao.txt", force=True)
    download(config.AO_URL, raw / "ao.txt", force=True)
    print("\nDone. Next: python -m src.build_dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download the station catalogue too")
    main(parser.parse_args().force)
