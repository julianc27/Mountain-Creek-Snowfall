"""Train and honestly evaluate forecast models at each checkpoint.

For every checkpoint (Nov 1, Dec 1, ...) we ask: "Will this season finish ABOVE the median?"
Evaluation is leave-one-season-out cross-validation: each season is predicted by a model that never
saw it, and the "median" threshold is recomputed from the training seasons only (no leakage).

Every model is compared against climatology - always forecasting the long-run base rate (~50%).
Brier Skill Score (BSS) = 1 - Brier(model) / Brier(climatology):
    BSS > 0  -> better than just guessing the average
    BSS = 0  -> no better
    BSS < 0  -> worse (overfitting!)

Usage:  python -m src.train
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config

CLIMATE_FEATURES = ["oni", "nao_recent", "ao_recent", "temp_anom_to_date_f"]
SNOW_FEATURES = ["snow_to_date_in", "depth_days_to_date", "snowmaking_nights_to_date"]
MODELS = {
    "climate_indices_only": CLIMATE_FEATURES,
    "full": CLIMATE_FEATURES + SNOW_FEATURES,
}
MIN_SEASONS = 12


def make_classifier():
    # Strong regularisation (small C) because we only have a few dozen seasons.
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3))


def make_regressor():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=5.0))


def usable(cols: list[str], df: pd.DataFrame) -> list[str]:
    """Drop features that are entirely missing or constant at this checkpoint."""
    return [c for c in cols if df[c].notna().sum() >= 5 and df[c].nunique(dropna=True) > 1]


def loso_cv(df: pd.DataFrame, feats: list[str], target: str) -> pd.DataFrame:
    rows = []
    seasons = df["season"].to_numpy()
    for i, held_out in enumerate(seasons):
        train, test = df[df["season"] != held_out], df.iloc[[i]]
        thr = train[target].median()
        y_train = (train[target] > thr).astype(int)
        y_test = int(test[target].iloc[0] > thr)
        base_rate = y_train.mean()

        clf = make_classifier().fit(train[feats], y_train)
        reg = make_regressor().fit(train[feats], train[target])
        rows.append({
            "season": held_out,
            "actual": test[target].iloc[0],
            "above_median": y_test,
            "p_model": clf.predict_proba(test[feats])[0, 1],
            "p_clim": base_rate,
            "pred_total": reg.predict(test[feats])[0],
            "clim_total": train[target].mean(),
        })
    return pd.DataFrame(rows)


def score(pred: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    bs_model = (pred["p_model"] - pred["above_median"]) ** 2
    bs_clim = (pred["p_clim"] - pred["above_median"]) ** 2
    bss = 1 - bs_model.mean() / bs_clim.mean()

    # Bootstrap over seasons: how sure are we the skill isn't luck?
    rng = np.random.default_rng(seed)
    n = len(pred)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(1 - bs_model.iloc[idx].mean() / bs_clim.iloc[idx].mean())
    lo, hi = np.percentile(boots, [5, 95])

    return {
        "n_seasons": n,
        "brier_model": bs_model.mean(),
        "brier_climatology": bs_clim.mean(),
        "brier_skill_score": bss,
        "bss_90ci_low": lo,
        "bss_90ci_high": hi,
        "hit_rate": ((pred["p_model"] > 0.5).astype(int) == pred["above_median"]).mean(),
        "mae_model_in": (pred["pred_total"] - pred["actual"]).abs().mean(),
        "mae_climatology_in": (pred["clim_total"] - pred["actual"]).abs().mean(),
    }


def plot_skill(results: pd.DataFrame, path: Path) -> None:
    order = [config.checkpoint_label(cp) for cp in config.CHECKPOINTS]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, style in [("climate_indices_only", "o--"), ("full", "o-")]:
        r = results[results["model"] == name].set_index("checkpoint").reindex(order).dropna(subset=["brier_skill_score"])
        ax.plot(r.index, r["brier_skill_score"], style, label=name.replace("_", " "), lw=2)
        ax.fill_between(r.index, r["bss_90ci_low"], r["bss_90ci_high"], alpha=0.15)
    ax.axhline(0, color="grey", lw=1)
    ax.set_ylabel("Brier skill score vs climatology")
    ax.set_xlabel("Forecast issued on")
    ax.set_title(f"{config.RESORT_NAME}: how much better than 'average' is the forecast?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(processed_dir: Path = config.PROCESSED_DIR, models_dir: Path = config.MODELS_DIR,
         reports_dir: Path = config.REPORTS_DIR) -> pd.DataFrame:
    target = config.TARGET
    seasons = pd.read_csv(processed_dir / "seasons.csv")
    features = pd.read_csv(processed_dir / "features.csv")
    data = features.merge(seasons[["season", target]], on="season").dropna(subset=[target])

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    results, all_preds, final = [], [], {}

    for cp in config.CHECKPOINTS:
        label = config.checkpoint_label(cp)
        df = data[data["checkpoint"] == label].reset_index(drop=True)
        if len(df) < MIN_SEASONS:
            print(f"{label}: only {len(df)} seasons, skipping")
            continue
        final[label] = {}
        for name, cols in MODELS.items():
            feats = usable(cols, df)
            pred = loso_cv(df, feats, target)
            results.append({"checkpoint": label, "model": name, **score(pred)})
            all_preds.append(pred.assign(checkpoint=label, model=name))
            thr = df[target].median()
            final[label][name] = {
                "classifier": make_classifier().fit(df[feats], (df[target] > thr).astype(int)),
                "regressor": make_regressor().fit(df[feats], df[target]),
                "features": feats,
                "threshold": thr,
                "n_train": len(df),
            }

    results = pd.DataFrame(results)
    results.to_csv(reports_dir / "cv_results.csv", index=False)
    pd.concat(all_preds).to_csv(reports_dir / "cv_predictions.csv", index=False)
    joblib.dump({"target": target, "models": final}, models_dir / "models.joblib")
    plot_skill(results, reports_dir / "skill_by_checkpoint.png")

    pd.set_option("display.width", 140)
    show = results[["checkpoint", "model", "n_seasons", "brier_skill_score", "bss_90ci_low", "bss_90ci_high",
                    "hit_rate", "mae_model_in", "mae_climatology_in"]]
    print(f"\nTarget: {target} above median   (leave-one-season-out cross-validation)\n")
    print(show.round(3).to_string(index=False))
    print(f"\nSaved reports/cv_results.csv, reports/skill_by_checkpoint.png, models/models.joblib")
    print("Next: python -m src.predict   or   streamlit run app.py")
    return results


if __name__ == "__main__":
    main()
