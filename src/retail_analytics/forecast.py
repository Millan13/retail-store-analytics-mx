"""Forecasting models and the rolling-origin backtest.

Every model exposes the same signature, ``predict(train, test, features) -> array``:
``train`` holds rows with date <= origin and a known target; ``test`` holds the
``horizon`` days after the origin. Baselines ignore ``features``.

Validation follows the rolling forecasting origin of Hyndman & Athanasopoulos,
*Forecasting: Principles and Practice* (3rd ed., section 5.10): for each origin the
model is trained only on data up to the origin and evaluated on the next days.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from retail_analytics.features import CATEGORICAL_FEATURES, SERIES_KEYS

Model = Callable[[pd.DataFrame, pd.DataFrame, list[str]], np.ndarray]


# --------------------------------------------------------------------------- splits
def weekly_origins(start: str, end: str) -> list[pd.Timestamp]:
    """List the Sundays between ``start`` and ``end`` (inclusive) used as forecast origins.

    Parameters
    ----------
    start, end : str
        Dates in ``YYYY-MM-DD`` format; both must be Sundays.

    Returns
    -------
    list of pandas.Timestamp
    """
    origins = list(pd.date_range(start, end, freq="W-SUN"))
    if not origins or origins[0] != pd.Timestamp(start) or origins[-1] != pd.Timestamp(end):
        raise ValueError("start and end must be Sundays.")
    return origins


def split_at_origin(
    df: pd.DataFrame, origin: pd.Timestamp, horizon: int, target: str, train_start: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the modeling table at a forecast origin.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of ``features.build_features``.
    origin : pandas.Timestamp
        Last day of known data.
    horizon : int
        Days to forecast after the origin.
    target : str
        Target column; training rows with a null target are dropped.
    train_start : str
        First date used for training (enough history for the lags).

    Returns
    -------
    tuple of pandas.DataFrame
        ``(train, test)``. ``test`` keeps rows with a null target so every model
        predicts the same rows; they are excluded later when scoring.
    """
    train = df[(df["date"] >= train_start) & (df["date"] <= origin) & df[target].notna()]
    test = df[(df["date"] > origin) & (df["date"] <= origin + pd.Timedelta(days=horizon))]
    return train, test


# --------------------------------------------------------------------------- baselines
def seasonal_naive(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Forecast each day with the same weekday of the previous week (``lag_7``).

    When ``lag_7`` is missing (POS failure that day), fall back to the 4-week
    same-weekday mean so the baseline is not penalized for a data gap.
    """
    return test["lag_7"].fillna(test["same_weekday_mean_4w"]).to_numpy()


def moving_average(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], window: int = 7
) -> np.ndarray:
    """Forecast every day with the mean of the last ``window`` known days of the series.

    This reproduces the logic of the current replenishment signal (a 7-day moving
    average) but without its look-ahead: only data up to the origin is used.
    """
    origin = train["date"].max()
    recent = train[train["date"] > origin - pd.Timedelta(days=window)]
    level = recent.groupby(SERIES_KEYS, observed=True)["units_sold"].mean().rename("pred")
    return (
        test[SERIES_KEYS].merge(level.reset_index(), on=SERIES_KEYS, how="left")["pred"].to_numpy()
    )


def _holt_winters_one(y: pd.Series, horizon: int) -> np.ndarray:
    # Fill gaps with the same weekday of previous weeks (keeps the weekly pattern);
    # interpolation only as a last resort.
    y = y.fillna(y.shift(7)).fillna(y.shift(14)).interpolate(limit_direction="both")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(
            y.to_numpy(), trend="add", damped_trend=True, seasonal="mul", seasonal_periods=7
        ).fit(optimized=True)
    return fit.forecast(horizon)


def holt_winters(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], n_jobs: int = -1
) -> np.ndarray:
    """Fit one Holt-Winters model per series (damped trend, multiplicative weekly season).

    Classical per-series benchmark: it captures level, trend and the weekly cycle,
    but it cannot use calendar events or information shared across stores.
    """
    horizon = test["date"].nunique()
    # Reindex every series to the full training calendar so the forecast always starts
    # the day after the origin, even if the last days before it have no data.
    calendar = pd.date_range(train["date"].min(), train["date"].max(), freq="D")
    series = {
        k: g.set_index("date")["units_sold"].reindex(calendar)
        for k, g in train.groupby(SERIES_KEYS, observed=True)
    }
    preds = Parallel(n_jobs=n_jobs)(delayed(_holt_winters_one)(y, horizon) for y in series.values())
    days = np.sort(test["date"].unique())
    long = pd.DataFrame(
        [
            (*k, d, v)
            for k, p in zip(series, preds, strict=True)
            for d, v in zip(days, p, strict=True)
        ],
        columns=[*SERIES_KEYS, "date", "pred"],
    )
    return (
        test[[*SERIES_KEYS, "date"]]
        .merge(long, on=[*SERIES_KEYS, "date"], how="left")["pred"]
        .to_numpy()
    )


# --------------------------------------------------------------------------- learned models
def poisson_glm(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], alpha: float = 1e-4
) -> np.ndarray:
    """Poisson regression (log link) on the same features as the tree models.

    Imputation and scaling live inside the pipeline, so they are fitted on the
    training rows only. Sales-level features enter in log so that, with the log
    link, they act multiplicatively, as in the data.
    """
    cats = [c for c in features if c in CATEGORICAL_FEATURES]
    nums = [c for c in features if c not in cats]
    level_cols = [
        c
        for c in nums
        if c.startswith(("lag_", "roll_", "same_weekday", "store_tx")) and "was_" not in c
    ]

    def prep(d: pd.DataFrame) -> pd.DataFrame:
        x = d[features].copy()
        x[level_cols] = np.log1p(x[level_cols].astype(float))
        x[cats] = x[cats].astype(str)
        return x

    model = make_pipeline(
        ColumnTransformer(
            [
                ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), nums),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
            ]
        ),
        PoissonRegressor(alpha=alpha, max_iter=1000),
    )
    model.fit(prep(train), train["units_sold"])
    return model.predict(prep(test))


def lightgbm(train: pd.DataFrame, test: pd.DataFrame, features: list[str], **params) -> np.ndarray:
    """Global LightGBM model across all series; missing values handled natively."""
    defaults = dict(
        objective="poisson",
        learning_rate=0.05,
        n_estimators=500,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model = lgb.LGBMRegressor(**{**defaults, **params})
    model.fit(
        train[features],
        train["units_sold"],
        categorical_feature=[c for c in features if c in CATEGORICAL_FEATURES],
    )
    return model.predict(test[features])


def xgboost(train: pd.DataFrame, test: pd.DataFrame, features: list[str], **params) -> np.ndarray:
    """Global XGBoost model with native categorical support (``hist`` tree method)."""
    defaults = dict(
        objective="count:poisson",
        learning_rate=0.05,
        n_estimators=500,
        max_depth=8,
        min_child_weight=50,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        random_state=42,
    )
    model = xgb.XGBRegressor(**{**defaults, **params})
    model.fit(train[features], train["units_sold"])
    return model.predict(test[features])


# --------------------------------------------------------------------------- backtest
def backtest(
    df: pd.DataFrame,
    model: Model,
    origins: list[pd.Timestamp],
    features: list[str],
    horizon: int,
    train_start: str,
    target: str = "units_sold",
) -> pd.DataFrame:
    """Run a model over several weekly origins (rolling forecasting origin).

    Parameters
    ----------
    df : pandas.DataFrame
        Output of ``features.build_features``.
    model : callable
        ``model(train, test, features) -> predictions``.
    origins : list of pandas.Timestamp
        Forecast origins (Sundays).
    features : list of str
        Feature columns passed to the model.
    horizon : int
        Days forecast after each origin.
    train_start : str
        First training date.
    target : str
        Target column.

    Returns
    -------
    pandas.DataFrame
        One row per forecast: keys, ``origin``, ``y_true`` (may be null) and ``y_pred``.
    """
    out = []
    for origin in origins:
        train, test = split_at_origin(df, origin, horizon, target, train_start)
        pred = np.clip(model(train, test, features), 0, None)
        out.append(
            test[[*SERIES_KEYS, "date"]].assign(
                origin=origin, y_true=test[target].to_numpy(), y_pred=pred
            )
        )
    return pd.concat(out, ignore_index=True)
