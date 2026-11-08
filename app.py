"""Streamlit dashboard:  streamlit run app.py"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.predict import current_forecast

st.set_page_config(page_title=f"{config.RESORT_NAME} Season Forecast", page_icon="🏂", layout="wide")

proc, reports = config.PROCESSED_DIR, config.REPORTS_DIR
needed = [proc / "seasons.csv", proc / "daily.csv", reports / "cv_results.csv", config.MODELS_DIR / "models.joblib"]
if not all(p.exists() for p in needed):
    st.error("No data yet. Run:  `python -m src.fetch_data && python -m src.build_dataset && python -m src.train`")
    st.stop()


@st.cache_data
def load():
    seasons = pd.read_csv(proc / "seasons.csv")
    daily = pd.read_csv(proc / "daily.csv", parse_dates=["date"])
    cv = pd.read_csv(reports / "cv_results.csv")
    return seasons, daily, cv


seasons, daily, cv = load()
target = config.TARGET
done = seasons.dropna(subset=[target])
units = {"snowfall_in": "in", "depth_days": "days", "snowmaking_nights": "nights"}[target]

st.title(f"🏂 {config.RESORT_NAME} season forecast")
st.caption(f"{config.RESORT_LOCATION} · target: **{target.replace('_', ' ')}** · "
           f"{len(done)} seasons of NOAA station data ({done['season'].min()}–{done['season'].max()})")

# ---- This season -----------------------------------------------------------------------
try:
    fc = current_forecast()
except RuntimeError as e:
    fc = pd.DataFrame()
    st.info(str(e))

if not fc.empty:
    latest = fc.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Chance of an above-median {latest['season']} season", f"{latest['p_above_median']:.0%}")
    c2.metric(f"Expected total ({units})", f"{latest['expected_total']:.0f}",
              f"{latest['expected_total'] - latest['median_season']:+.0f} vs median")
    c3.metric("Forecast issued", latest["checkpoint"])
    if len(fc) > 1:
        st.line_chart(fc.set_index("checkpoint")["p_above_median"], y_label="P(above median)")

# ---- Season so far vs history ----------------------------------------------------------
if target == "snowfall_in":
    st.subheader("Cumulative snowfall: this season vs. history")
    d = daily.copy()
    d["day"] = (d["date"] - pd.to_datetime(d["season"].astype(str) + "-10-01")).dt.days
    d["cum"] = d.groupby("season")["snow_in"].cumsum()
    wide = d.pivot_table(index="day", columns="season", values="cum")
    hist = wide[[c for c in wide.columns if c in set(done["season"])]]
    band = pd.DataFrame({
        "10th pct": hist.quantile(0.1, axis=1),
        "median": hist.median(axis=1),
        "90th pct": hist.quantile(0.9, axis=1),
    })
    live = seasons.loc[seasons["in_progress"], "season"]
    if len(live) and int(live.iloc[0]) in wide.columns:
        band["this season"] = wide[int(live.iloc[0])]
    band.index = pd.Timestamp("2000-10-01") + pd.to_timedelta(band.index, unit="D")
    st.line_chart(band, y_label="inches since Oct 1")

# ---- History ---------------------------------------------------------------------------
st.subheader(f"Every season on record ({units})")
hist_tbl = done.set_index("label")[target]
st.bar_chart(hist_tbl)

# ---- Model skill -----------------------------------------------------------------------
st.subheader("How good is the forecast? (leave-one-season-out cross-validation)")
st.markdown(
    "**Brier skill score** compares the model to always guessing the long-run average. "
    "Above 0 means it adds information; the shaded range in `reports/skill_by_checkpoint.png` "
    "shows the 90% bootstrap interval."
)
order = [config.checkpoint_label(cp) for cp in config.CHECKPOINTS]
skill = cv.pivot(index="checkpoint", columns="model", values="brier_skill_score").reindex(order).dropna(how="all")
st.line_chart(skill, y_label="Brier skill score")
st.dataframe(cv.round(3), hide_index=True)
