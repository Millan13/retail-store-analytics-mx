"""End-to-end forecast experiment: selection and tuning on validation, comparison on test.

Run with ``python -m retail_analytics.pipeline`` (or ``make forecast``). Results are
written to ``data/processed/`` and every backtest is logged to MLflow (``mlflow.db``;
browse with ``uv run mlflow ui --backend-store-uri sqlite:///mlflow.db``).

Protocol (decided before looking at test results):

1. Feature selection by group on the validation origins: a group is kept only if
   removing it worsens validation WAPE by more than ``MIN_GAIN``.
2. Small hyperparameter grids on the validation origins.
3. One final run of every model on the test origins with the chosen setup.
"""

from __future__ import annotations

import json
import logging
import os
import time
from functools import partial

import mlflow
import pandas as pd

from retail_analytics import evaluate as ev
from retail_analytics import forecast as fc
from retail_analytics.cleaning import build_clean_panel
from retail_analytics.config import PROJECT_ROOT, load_config, project_path
from retail_analytics.features import FEATURE_GROUPS, build_features, feature_columns
from retail_analytics.io import load_raw

log = logging.getLogger(__name__)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# A feature group must improve validation WAPE by at least 0.1 points to stay.
MIN_GAIN = 0.001

LGBM_GRID = [
    {"objective": obj, "num_leaves": leaves, "n_estimators": n}
    for obj in ("poisson", "tweedie")
    for leaves in (31, 63)
    for n in (300, 600)
]
XGB_GRID = [{"max_depth": d, "n_estimators": n} for d in (6, 8) for n in (300, 600)]
GLM_GRID = [{"alpha": a} for a in (1e-4, 1e-2)]


def _run(df, name, model, origins, features, cfg, stage, params=None) -> pd.DataFrame:
    """Backtest one model configuration and log it to MLflow."""
    start = time.time()
    preds = fc.backtest(df, model, origins, features, cfg["horizon"], cfg["train_start"])
    s = ev.score(preds).iloc[0]
    with mlflow.start_run(run_name=f"{stage}-{name}"):
        mlflow.log_params(
            {"model": name, "stage": stage, "n_features": len(features), **(params or {})}
        )
        mlflow.log_metrics({"wape": s["wape"], "bias": s["bias"], "seconds": time.time() - start})
    log.info("%s %-10s %s wape=%.4f bias=%+.4f", stage, name, params or "", s["wape"], s["bias"])
    return preds.assign(model=name)


def select_feature_groups(df, origins, cfg) -> pd.DataFrame:
    """Group-wise ablation with LightGBM: validation WAPE without each group."""
    full = feature_columns()
    rows = [
        {
            "grupo_quitado": "(ninguno)",
            "wape": ev.wape(*_yy(_run(df, "lgbm", fc.lightgbm, origins, full, cfg, "ablation"))),
        }
    ]
    for group in FEATURE_GROUPS:
        feats = feature_columns([g for g in FEATURE_GROUPS if g != group])
        preds = _run(df, f"lgbm-sin-{group}", fc.lightgbm, origins, feats, cfg, "ablation")
        rows.append({"grupo_quitado": group, "wape": ev.wape(*_yy(preds))})
    out = pd.DataFrame(rows)
    out["cambio_vs_completo"] = out["wape"] - out.loc[0, "wape"]
    out["se_queda"] = out["cambio_vs_completo"] > MIN_GAIN
    out.loc[0, "se_queda"] = pd.NA
    return out


def _yy(preds: pd.DataFrame):
    return preds["y_true"].to_numpy(), preds["y_pred"].to_numpy()


def tune(df, origins, features, cfg) -> tuple[dict, pd.DataFrame]:
    """Pick the best configuration of each learned model on the validation origins."""
    results, best = [], {}
    for name, model, grid in [
        ("poisson_glm", fc.poisson_glm, GLM_GRID),
        ("lightgbm", fc.lightgbm, LGBM_GRID),
        ("xgboost", fc.xgboost, XGB_GRID),
    ]:
        for params in grid:
            preds = _run(
                df, name, partial(model, **params), origins, features, cfg, "tuning", params
            )
            results.append(
                {"modelo": name, "parametros": json.dumps(params), "wape": ev.wape(*_yy(preds))}
            )
        table = pd.DataFrame([r for r in results if r["modelo"] == name])
        best[name] = json.loads(table.loc[table["wape"].idxmin(), "parametros"])
    return best, pd.DataFrame(results)


def main() -> None:
    """Run the full protocol and save the outputs used by ``02_pronostico.ipynb``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config()["forecast"]
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("demand-forecast")
    out = project_path("processed")

    raw = load_raw()
    panel = build_clean_panel(raw.transactions, raw.stores, raw.calendar)
    df = build_features(panel, cfg["horizon"])
    val_origins = fc.weekly_origins(*cfg["validation_origins"])
    test_origins = fc.weekly_origins(*cfg["test_origins"])

    # 1. Feature selection (validation only)
    ablation = select_feature_groups(df, val_origins, cfg)
    ablation.to_csv(out / "validation_ablation.csv", index=False)
    kept = ablation.loc[ablation["se_queda"].fillna(False).astype(bool), "grupo_quitado"].tolist()
    # Leave-one-group-out can drop several redundant groups at once (e.g. lags and
    # rolling means). Check the reduced set against the full set before accepting it.
    full_wape = ablation.loc[0, "wape"]
    reduced = _run(
        df, "lgbm-seleccion", fc.lightgbm, val_origins, feature_columns(kept), cfg, "ablation"
    )
    reduced_wape = ev.wape(*_yy(reduced))
    candidate = list(kept)
    if not kept or reduced_wape > full_wape + MIN_GAIN:
        log.info(
            "Reduced set worse than full (%.4f vs %.4f): keeping all groups",
            reduced_wape,
            full_wape,
        )
        kept = list(FEATURE_GROUPS)
    features = feature_columns(kept)
    pd.DataFrame(
        {
            "conjunto": ["completo", "candidato (ablación)", "elegido"],
            "wape": [
                full_wape,
                reduced_wape,
                full_wape if kept == list(FEATURE_GROUPS) else reduced_wape,
            ],
            "grupos": [", ".join(FEATURE_GROUPS), ", ".join(candidate), ", ".join(kept)],
        }
    ).to_csv(out / "validation_selection_check.csv", index=False)
    log.info("Selected groups: %s", kept)

    # 2. Hyperparameters (validation only)
    best, tuning = tune(df, val_origins, features, cfg)
    tuning.to_csv(out / "validation_tuning.csv", index=False)

    # 3. Final comparison on the test origins
    models = {
        "naive_estacional": fc.seasonal_naive,
        "sistema_actual": fc.moving_average,
        "holt_winters": fc.holt_winters,
        "poisson_glm": partial(fc.poisson_glm, **best["poisson_glm"]),
        "lightgbm": partial(fc.lightgbm, **best["lightgbm"]),
        "xgboost": partial(fc.xgboost, **best["xgboost"]),
    }
    preds = pd.concat(
        [
            _run(df, name, m, test_origins, features, cfg, "test", best.get(name))
            for name, m in models.items()
        ],
        ignore_index=True,
    )
    preds.to_parquet(out / "test_predictions.parquet", index=False)

    prices = panel[["store_id", "category", "date"]].assign(
        unit_price=panel["amount_total"] / panel["units_sold"]
    )
    prices.to_parquet(out / "unit_prices.parquet", index=False)
    (out / "forecast_setup.json").write_text(
        json.dumps({"feature_groups": kept, "features": features, "best_params": best}, indent=2),
        encoding="utf-8",
    )
    log.info("Done. Outputs in %s", out)


if __name__ == "__main__":
    main()
