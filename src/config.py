"""Project-wide settings. Change these to point the project at a different mountain."""
from pathlib import Path

# --- Resort -----------------------------------------------------------------
RESORT_NAME = "Mountain Creek"
RESORT_LOCATION = "Vernon, NJ"
RESORT_LAT = 41.1809
RESORT_LON = -74.5029

# --- Station search ---------------------------------------------------------
SEARCH_RADIUS_KM = 45        # look for NOAA stations within this distance
MIN_SNOW_YEARS = 25          # station must have at least this many years of snowfall records
MAX_STATIONS = 6             # keep the N nearest qualifying stations
MIN_LAST_YEAR_LAG = 3        # station must still be reporting within this many years

# --- Season definition ------------------------------------------------------
# A "season" is labelled by the year it starts in: season 2025 = Nov 2025 - Apr 2026.
SEASON_START = (11, 1)       # (month, day)
SEASON_END = (4, 30)
MIN_COMPLETENESS = 0.85      # fraction of season days that must have a snowfall report
DEPTH_DAY_THRESHOLD_IN = 1.0 # a "snow-on-ground day" has at least this much depth
SNOWMAKING_TMIN_F = 25.0     # nights at/below this temp count as good snowmaking nights

# Forecast checkpoints (month, day). The model only sees data available before each date.
CHECKPOINTS = [(11, 1), (12, 1), (12, 15), (1, 1), (1, 15), (2, 1)]

# What counts as a "good season"? Options: "snowfall_in", "depth_days", "snowmaking_nights"
TARGET = "snowfall_in"

# --- Paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# --- Data sources (all free, public NOAA data) ------------------------------
GHCN_STATIONS_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
GHCN_INVENTORY_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"
GHCN_STATION_CSV_URL = (
    "https://www.ncei.noaa.gov/data/global-historical-climatology-network/daily/access/{station_id}.csv"
)
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
NAO_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii"
AO_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii"


def checkpoint_label(cp: tuple) -> str:
    month, day = cp
    names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{names[month]} {day}"
