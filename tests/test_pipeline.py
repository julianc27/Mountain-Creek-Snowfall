"""End-to-end test on synthetic data:  python -m pytest -q"""
import datetime as dt

import pandas as pd
import pytest

from src import build_dataset, config, predict, train
from tests.synthetic import make_raw


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    raw, proc, models, reports = root / "raw", root / "processed", root / "models", root / "reports"
    make_raw(raw)
    build_dataset.main(raw, proc, today=dt.date(2026, 11, 7))
    results = train.main(proc, models, reports)
    return {"proc": proc, "models": models, "reports": reports, "results": results}


def test_units_converted(project):
    seasons = pd.read_csv(project["proc"] / "seasons.csv")
    done = seasons.dropna(subset=["snowfall_in"])
    assert 5 < done["snowfall_in"].median() < 150  # inches, not millimetres


def test_gap_seasons_filled_from_neighbor(project):
    seasons = pd.read_csv(project["proc"] / "seasons.csv").set_index("season")
    assert seasons.loc[1989, "filled_from_neighbor"]
    assert not pd.isna(seasons.loc[1989, "snowfall_in"])


def test_current_season_has_no_target(project):
    seasons = pd.read_csv(project["proc"] / "seasons.csv").set_index("season")
    assert seasons.loc[2026, "in_progress"]
    assert pd.isna(seasons.loc[2026, "snowfall_in"])


def test_no_future_checkpoints(project):
    feats = pd.read_csv(project["proc"] / "features.csv")
    current = feats[feats["season"] == 2026]
    assert current["checkpoint"].tolist() == ["Nov 1"]  # Dec 1 hasn't happened on Nov 7


def test_snow_to_date_uses_only_past_data(project):
    feats = pd.read_csv(project["proc"] / "features.csv")
    f = feats[feats["checkpoint"] == "Dec 1"]
    f15 = feats[feats["checkpoint"] == "Dec 15"]
    m = f.merge(f15, on="season", suffixes=("_1", "_15")).dropna(subset=["snow_to_date_in_1", "snow_to_date_in_15"])
    assert (m["snow_to_date_in_15"] >= m["snow_to_date_in_1"] - 1e-9).all()


def test_in_season_model_beats_climatology(project):
    r = project["results"]
    late = r[(r["model"] == "full") & (r["checkpoint"] == "Feb 1")]
    assert late["brier_skill_score"].iloc[0] > 0.1


def test_predict_current_season(project):
    fc = predict.current_forecast(project["proc"], project["models"])
    assert len(fc) == 1 and 0 <= fc["p_above_median"].iloc[0] <= 1


def test_model_has_version(project):
    fc = predict.current_forecast(project["proc"], project["models"])
    v = fc["model_version"].iloc[0]
    assert isinstance(v, str) and len(v) == 12 and v != "unversioned"


def test_forecast_log_is_append_only(tmp_path, project):
    log = tmp_path / "forecast_log.csv"
    fc = predict.current_forecast(project["proc"], project["models"])

    first = predict.log_forecast(fc, log, issued_at=dt.datetime(2026, 11, 3, tzinfo=dt.timezone.utc))
    assert first["checkpoint"].tolist() == ["Nov 1"]
    snapshot = log.read_bytes()

    # Re-running with revised numbers for an already-logged checkpoint must not touch the log.
    revised = fc.assign(p_above_median=0.99)
    assert predict.log_forecast(revised, log).empty
    assert log.read_bytes() == snapshot

    # A new checkpoint is appended; the original line stays byte-for-byte identical.
    dec1 = fc.assign(checkpoint="Dec 1", p_above_median=0.42)
    added = predict.log_forecast(pd.concat([revised, dec1]), log)
    assert added["checkpoint"].tolist() == ["Dec 1"]
    assert log.read_bytes().startswith(snapshot)

    out = pd.read_csv(log).set_index("checkpoint")
    assert out.loc["Nov 1", "p_above_median"] == pytest.approx(fc["p_above_median"].iloc[0])
    assert out.loc["Nov 1", "issued_at"].startswith("2026-11-03")
    assert out.loc["Dec 1", "p_above_median"] == pytest.approx(0.42)


def test_train_refuses_to_replace_model_mid_season(tmp_path, project):
    reports = tmp_path / "reports"
    fc = predict.current_forecast(project["proc"], project["models"])
    predict.log_forecast(fc, reports / train.FORECAST_LOG)
    with pytest.raises(RuntimeError, match="--force"):
        train.main(project["proc"], tmp_path / "models", reports)


def test_by_station_long_format_converted_for_loader(tmp_path):
    """NOAA's by_station files are long format (one row per element); convert to the loader's wide layout."""
    from src.fetch_data import ghcn_long_to_wide

    long_csv = (
        "USC00000001,20240105,SNOW,51,,,7,0700\n"
        "USC00000001,20240105,TMAX,-50,,,7,0700\n"
        "USC00000001,20240105,PRCP,10,,,7,0700\n"
        "USC00000001,20240106,SNOW,999,,X,7,0700\n"   # failed quality check
        "USC00000001,20240106,TMIN,-100,,,7,\n"
    )
    src = tmp_path / "long.csv"
    src.write_text(long_csv)
    dest = tmp_path / "wide.csv"
    ghcn_long_to_wide(src, dest)

    daily = build_dataset.load_station(dest)
    assert daily.loc["2024-01-05", "snow_in"] == pytest.approx(51 / 25.4)
    assert daily.loc["2024-01-05", "tmax_f"] == pytest.approx(23.0)
    assert pd.isna(daily.loc["2024-01-06", "snow_in"])           # QC-failed value masked
    assert daily.loc["2024-01-06", "tmin_f"] == pytest.approx(14.0)
