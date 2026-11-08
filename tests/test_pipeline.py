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
