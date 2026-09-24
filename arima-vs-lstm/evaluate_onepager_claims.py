"""Re-run the numbers in ARIMA_vs_LSTM_OnePager.md.

Two evaluations are run on final_4year.csv:

1. Literal: the protocol of ARIMA_vs_LSTM_Comparison.ipynb, which the
   one-pager describes. Chronological 90/10 split, SARIMA(1,1,1)x(1,1,1,24)
   fitted on the training part and forecasting the whole test window in one go,
   against a Keras LSTM(50) that predicts each test hour from the actual
   previous 60 hours. The one-pager's per-period table (last 168 test hours)
   and error-distribution table are computed from these predictions.

2. Like for like: both models forecast 1 to 24 hours ahead from the same
   origins, every hour of the test window, using only data up to the origin.
   ARIMA keeps its training-set parameters and its Kalman state is updated
   with each observed hour (a variant also re-estimates the parameters at the
   start of each month). The LSTM gets the actual 60 hours up to the origin
   and feeds its own predictions back for horizons beyond 1 hour, as the
   notebook's 24-hour forecast does. The one-pager's per-horizon table (MAE for
   horizons 1-6, 7-12, 13-18, 19-24) needs forecasts with known outcomes, so it
   is computed here.

The notebook's own 24-hour forecast starts after the last hour of data and
cannot be scored.

Usage (from the repository root):

    DATA_DIR=/abs/path/to/data python arima-vs-lstm/evaluate_onepager_claims.py
    python arima-vs-lstm/evaluate_onepager_claims.py --data-dir data --seeds 42,43,44

Outputs go to arima-vs-lstm/results/ (onepager_claims.json, onepager_claims.md).
The helpers at the top only need numpy and pandas and are tested offline in
tests/test_evaluate_onepager_claims.py; statsmodels, pmdarima and TensorFlow
are imported when the models run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = "Hourly Demand Met (in MW)"
DATA_FILE = "final_4year.csv"

SPLIT_RATIO = 0.90
LOOKBACK = 60
MAX_HORIZON = 24
REPORT_HORIZONS = (1, 6, 24)
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 24)
LSTM_UNITS = 50
LSTM_EPOCHS = 20
LSTM_BATCH = 32
LSTM_VAL_SPLIT = 0.1
LSTM_PATIENCE = 5
ENSEMBLE_WEIGHTS = (0.6, 0.4)  # ARIMA, LSTM, as in the one-pager's recommendation
# Hour-of-day bands, [start, end): the 10:00-14:00 band is hours 10, 11, 12 and 13.
PERIODS = {
    "Peak Hours (10:00-14:00)": (10, 14),
    "Evening Peak (18:00-22:00)": (18, 22),
    "Off-Peak (00:00-06:00)": (0, 6),
}
HORIZON_BANDS = {"1-6 hours": (1, 6), "7-12 hours": (7, 12), "13-18 hours": (13, 18), "19-24 hours": (19, 24)}
SAMPLE_WINDOW = 168  # the notebook plots the last 7 days of the test window

# What the one-pager states.
CLAIMS = {
    "counts": {"total": 35040, "train": 31536, "test": 3504},
    "headline": {
        "ARIMA": {"rmse": 341.28, "mae": 268.15, "mape": 4.82},
        "LSTM": {"rmse": 357.42, "mae": 279.83, "mape": 5.13},
        "arima_better_pct": {"rmse": 4.5, "mae": 4.2, "mape": 6.0},
    },
    "periods_rmse": {
        "Peak Hours (10:00-14:00)": {"ARIMA": 312.4, "LSTM": 341.8},
        "Evening Peak (18:00-22:00)": {"ARIMA": 328.7, "LSTM": 352.3},
        "Off-Peak (00:00-06:00)": {"ARIMA": 287.3, "LSTM": 298.6},
    },
    "error_distribution": {
        "ARIMA": {"mean_error": -12.3, "sd": 339.8, "band95": 666.0},
        "LSTM": {"mean_error": -8.7, "sd": 356.1, "band95": 697.9},
    },
    "horizon_bands_mae": {
        "1-6 hours": {"ARIMA": 198.3, "LSTM": 212.7, "arima_better_pct": 7.3},
        "7-12 hours": {"ARIMA": 245.6, "LSTM": 258.9, "arima_better_pct": 5.4},
        "13-18 hours": {"ARIMA": 289.4, "LSTM": 301.2, "arima_better_pct": 4.1},
        "19-24 hours": {"ARIMA": 312.8, "LSTM": 327.5, "arima_better_pct": 4.7},
    },
    "degradation_pct": {"ARIMA": 57.7, "LSTM": 54.0},
    "compute_minutes": {"ARIMA": 5, "LSTM": 10},
}


# ---------------------------------------------------------------------------
# Helpers (numpy/pandas only)
# ---------------------------------------------------------------------------

def load_series(path) -> pd.Series:
    """Hourly demand series, prepared as in the notebook (sorted, hourly, gaps interpolated)."""
    df = pd.read_csv(path, usecols=["Date and Time", TARGET])
    df["Date and Time"] = pd.to_datetime(df["Date and Time"], errors="coerce")
    df = df.dropna(subset=["Date and Time"]).set_index("Date and Time").sort_index()
    s = df[TARGET].astype(float).replace([np.inf, -np.inf], np.nan)
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(full).interpolate(method="time").ffill().bfill()
    s.name = TARGET
    return s


def chronological_split(n: int, ratio: float = SPLIT_RATIO) -> tuple[int, int]:
    """Train and test sizes of the notebook's split: train = int(n * ratio)."""
    n_train = int(n * ratio)
    return n_train, n - n_train


def drop_leap_days(s: pd.Series) -> pd.Series:
    """Remove every 29 February (24 rows per leap year)."""
    idx = s.index
    return s[~((idx.month == 2) & (idx.day == 29))]


def regression_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, MAPE (%) and the error distribution, with error = actual - predicted."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    err = y_true - y_pred
    sd = float(np.std(err, ddof=1)) if err.size > 1 else float("nan")
    nz = y_true != 0
    return {
        "n": int(err.size),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(np.abs(err[nz] / y_true[nz])) * 100),
        "mean_error": float(np.mean(err)),
        "sd": sd,
        "band95": 1.96 * sd,
    }


def pct_better(a: float, b: float, base: str = "b") -> float:
    """How much lower a is than b, in % of b (base="b") or of a (base="a").

    The one-pager's headline uses (LSTM - ARIMA) / LSTM; its horizon table uses
    (LSTM - ARIMA) / ARIMA. Positive values mean a (ARIMA) is lower.
    """
    denom = b if base == "b" else a
    return float((b - a) / denom * 100)


def degradation_pct(first: float, last: float) -> float:
    """Growth of error from the first to the last horizon band, in %."""
    return float((last / first - 1) * 100)


def hour_mask(index: pd.DatetimeIndex, start: int, end: int) -> np.ndarray:
    """Rows whose hour of day is in [start, end)."""
    h = np.asarray(index.hour)
    return (h >= start) & (h < end)


def period_rmse(index, y_true, y_pred, periods=PERIODS) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {}
    for name, (a, b) in periods.items():
        m = hour_mask(pd.DatetimeIndex(index), a, b)
        out[name] = {"n": int(m.sum()), "rmse": float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))}
    return out


def rolling_origins(n_train: int, n: int, max_h: int = MAX_HORIZON, step: int = 1) -> np.ndarray:
    """Positions t of the last observed hour, from the end of training to the last
    origin whose max_h-hour forecast still lies inside the data."""
    return np.arange(n_train - 1, n - max_h, step)


def target_matrix(y, origins, max_h: int = MAX_HORIZON) -> np.ndarray:
    """Actual values y[t + h] for each origin t (rows) and horizon h = 1..max_h (columns)."""
    y = np.asarray(y, dtype=float)
    return y[np.asarray(origins)[:, None] + np.arange(1, max_h + 1)[None, :]]


def persistence_forecasts(y, origins, max_h: int = MAX_HORIZON) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.repeat(y[np.asarray(origins)][:, None], max_h, axis=1)


def seasonal_naive_forecasts(y, origins, max_h: int = MAX_HORIZON, season: int = 24) -> np.ndarray:
    """Same hour of the previous day: y[t + h - season], known at the origin for h <= season."""
    if max_h > season:
        raise ValueError("seasonal naive needs max_h <= season")
    y = np.asarray(y, dtype=float)
    return y[np.asarray(origins)[:, None] + np.arange(1, max_h + 1)[None, :] - season]


def state_space_forecasts(predicted_state, design, transition, origins, max_h: int = MAX_HORIZON,
                          obs_intercept=None, state_intercept=None) -> np.ndarray:
    """h-step forecasts from a time-invariant linear Gaussian state-space model.

    predicted_state[:, t + 1] is a(t+1 | t), the state predicted from data up to t,
    as stored by statsmodels' Kalman filter. Returns an array (len(origins), max_h).
    """
    Z = np.asarray(design, dtype=float).reshape(1, -1)
    T = np.asarray(transition, dtype=float)
    k = T.shape[0]
    d = 0.0 if obs_intercept is None else float(np.asarray(obs_intercept).ravel()[0])
    c = np.zeros((k, 1)) if state_intercept is None else np.asarray(state_intercept, dtype=float).reshape(k, 1)
    a = np.asarray(predicted_state)[:, np.asarray(origins) + 1]
    out = np.empty((a.shape[1], max_h))
    for h in range(max_h):
        out[:, h] = (Z @ a).ravel() + d
        a = T @ a + c
    return out


def band_means(per_horizon, bands=HORIZON_BANDS) -> dict:
    """Average of a per-horizon metric (index 0 = horizon 1) over each horizon band."""
    v = np.asarray(per_horizon, dtype=float)
    return {name: float(v[a - 1:b].mean()) for name, (a, b) in bands.items()}


def diebold_mariano(e1, e2, h: int = 1, lag: int | None = None) -> dict:
    """Diebold-Mariano test on squared-error loss with a Bartlett (Newey-West) long-run
    variance and the Harvey-Leybourne-Newbold small-sample correction.

    Negative statistics mean model 1 has the lower loss. lag defaults to h - 1.
    """
    from scipy import stats

    d = np.asarray(e1, dtype=float) ** 2 - np.asarray(e2, dtype=float) ** 2
    n = d.size
    lag = h - 1 if lag is None else lag
    dc = d - d.mean()
    lrv = np.dot(dc, dc) / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * np.dot(dc[k:], dc[:-k]) / n
    if lrv <= 0:
        return {"stat": float("nan"), "p_value": float("nan"), "lag": lag, "n": n}
    stat = d.mean() / math.sqrt(lrv / n)
    stat *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {"stat": float(stat), "p_value": float(p), "lag": int(lag), "n": int(n)}


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def blas_single_thread():
    """Kalman filtering uses many small matrix products; one BLAS thread is faster."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1, user_api="blas")
    except ImportError:  # pragma: no cover
        import contextlib
        return contextlib.nullcontext()


def sarimax_kwargs(order, seasonal_order) -> dict:
    # The notebook's statsmodels path.
    return dict(order=tuple(order), seasonal_order=tuple(seasonal_order),
                enforce_stationarity=False, enforce_invertibility=False)


def fit_sarima(y_train, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    t0 = time.time()
    with blas_single_thread():
        res = SARIMAX(np.asarray(y_train, dtype=float), **sarimax_kwargs(order, seasonal_order)).fit(
            disp=False, maxiter=100, low_memory=True)  # low_memory: no smoother output (needs ~20 GB here)
    info = {
        "order": list(order), "seasonal_order": list(seasonal_order),
        "params": dict(zip(res.model.param_names, map(float, res.params))),
        "aic": float(res.aic), "converged": bool(res.mle_retvals.get("converged", False)),
        "iterations": int(res.mle_retvals.get("iterations", -1)), "n_train": int(res.nobs),
        "fit_seconds": round(time.time() - t0, 1),
    }
    return res, info


def sarima_rolling_forecasts(y, params, origins, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER,
                             max_h: int = MAX_HORIZON) -> np.ndarray:
    """Forecasts from each origin with fixed parameters and the Kalman state updated
    with every observation up to the origin."""
    from statsmodels.tsa.statespace import kalman_filter as kf
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Keep the predicted state means only; the covariance arrays would need several GB.
    keep_means = (kf.MEMORY_NO_FORECAST_COV | kf.MEMORY_NO_PREDICTED_COV | kf.MEMORY_NO_FILTERED
                  | kf.MEMORY_NO_GAIN | kf.MEMORY_NO_SMOOTHING | kf.MEMORY_NO_STD_FORECAST)
    with blas_single_thread():
        res = SARIMAX(np.asarray(y, dtype=float), **sarimax_kwargs(order, seasonal_order)).filter(
            params, conserve_memory=keep_means)
    ssm = res.model.ssm
    return state_space_forecasts(res.filter_results.predicted_state, ssm["design"], ssm["transition"],
                                 origins, max_h, ssm["obs_intercept"], ssm["state_intercept"])


def check_state_space_forecasts(y, params, origins, got, order, seasonal_order) -> float:
    """Largest gap between our forecasts and statsmodels' own forecast() for a few origins."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    gaps = []
    for i in (0, len(origins) // 2, len(origins) - 1):
        t = int(origins[i])
        with blas_single_thread():
            ref = SARIMAX(np.asarray(y[: t + 1], dtype=float), **sarimax_kwargs(order, seasonal_order)).filter(
                params, low_memory=True).forecast(got.shape[1])
        gaps.append(float(np.max(np.abs(np.asarray(ref) - got[i]))))
    return max(gaps)


def capped_arima_search(y_train, hours: int) -> dict:
    """The notebook's pmdarima auto_arima call (cell 15), on the last `hours` training hours."""
    import pmdarima as pm

    t0 = time.time()
    with blas_single_thread():
        model = pm.auto_arima(
            pd.Series(np.asarray(y_train, dtype=float)[-hours:]),
            start_p=1, max_p=3, start_q=1, max_q=3, d=None, max_d=2, seasonal=True, m=24,
            start_P=0, max_P=1, start_Q=0, max_Q=1, D=None, max_D=1, trace=False, error_action="ignore",
            suppress_warnings=True, stepwise=True, n_fits=20, max_iter=100, return_valid_fits=False,
            information_criterion="aic", n_jobs=1)
    return {"hours": int(hours), "order": list(model.order), "seasonal_order": list(model.seasonal_order),
            "with_intercept": bool(model.with_intercept), "aic_on_subsample": float(model.aic()),
            "search_seconds": round(time.time() - t0, 1)}


def set_seeds(seed: int) -> None:
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def make_windows(scaled, lookback: int = LOOKBACK):
    scaled = np.asarray(scaled, dtype=float).ravel()
    X = np.lib.stride_tricks.sliding_window_view(scaled[:-1], lookback)
    return X[..., None].copy(), scaled[lookback:].copy()


def train_lstm(train_scaled, seed: int, verbose: int = 0):
    """The notebook's Keras model and training call."""
    import tensorflow as tf
    from tensorflow import keras

    set_seeds(seed)
    X, y = make_windows(train_scaled)
    model = keras.Sequential([keras.Input(shape=(LOOKBACK, 1)), keras.layers.LSTM(LSTM_UNITS),
                              keras.layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=LSTM_PATIENCE, restore_best_weights=True)
    t0 = time.time()
    hist = model.fit(X, y, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, validation_split=LSTM_VAL_SPLIT,
                     callbacks=[stop], verbose=verbose)
    val = hist.history["val_loss"]
    info = {"seed": seed, "epochs_run": len(val), "best_epoch": int(np.argmin(val)) + 1,
            "best_val_loss": float(np.min(val)), "train_seconds": round(time.time() - t0, 1),
            "n_windows": int(len(X)), "tensorflow": tf.__version__}
    return model, info


def lstm_recursive_forecasts(model, scaled, origins, max_h: int = MAX_HORIZON) -> np.ndarray:
    """Scaled forecasts: the actual LOOKBACK hours up to each origin, then the model's own
    predictions fed back for horizons beyond 1."""
    scaled = np.asarray(scaled, dtype=float).ravel()
    idx = np.asarray(origins)[:, None] + np.arange(-LOOKBACK + 1, 1)[None, :]
    window = scaled[idx].astype("float32")
    out = np.empty((len(origins), max_h))
    for h in range(max_h):
        pred = model.predict(window[..., None], batch_size=4096, verbose=0).ravel()
        out[:, h] = pred
        window = np.concatenate([window[:, 1:], pred[:, None].astype("float32")], axis=1)
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def to_json(obj, nd: int = 4, indent: int = 2) -> str:
    """JSON text with every float written with nd decimals (values below 0.01 in
    e-notation), so the file has a uniform, diff-friendly format."""
    def enc(o, level):
        pad, end = " " * indent * (level + 1), " " * indent * level
        if isinstance(o, dict):
            if not o:
                return "{}"
            items = [f"{pad}{json.dumps(str(k))}: {enc(v, level + 1)}" for k, v in o.items()]
            return "{\n" + ",\n".join(items) + "\n" + end + "}"
        if isinstance(o, (list, tuple)):
            if not o:
                return "[]"
            if not any(isinstance(v, (dict, list, tuple)) for v in o):
                return "[" + ", ".join(enc(v, level + 1) for v in o) + "]"
            return "[\n" + ",\n".join(pad + enc(v, level + 1) for v in o) + "\n" + end + "]"
        if o is None or isinstance(o, (bool, np.bool_)):
            return json.dumps(None if o is None else bool(o))
        if isinstance(o, (int, np.integer)):
            return str(int(o))
        if isinstance(o, (float, np.floating)):
            x = float(o)
            if not math.isfinite(x):
                return "null"
            return f"{x:.{nd}e}" if 0 < abs(x) < 0.01 else f"{x:.{nd}f}"
        return json.dumps(o)
    return enc(obj, 0) + "\n"


def literal_protocol(y, index, n_train, arima_res, lstm_pred_1step) -> dict:
    """The notebook's comparison: ARIMA one-shot over the test window, LSTM one step ahead."""
    test = y[n_train:]
    test_idx = index[n_train:]
    arima = np.asarray(arima_res.forecast(steps=len(test)), dtype=float)
    lstm = np.asarray(lstm_pred_1step, dtype=float)
    ens = ENSEMBLE_WEIGHTS[0] * arima + ENSEMBLE_WEIGHTS[1] * lstm
    m = {"ARIMA": regression_metrics(test, arima), "LSTM": regression_metrics(test, lstm),
         "Ensemble 60/40": regression_metrics(test, ens)}
    w = slice(len(test) - SAMPLE_WINDOW, len(test))
    out = {
        "test_window": [str(test_idx[0]), str(test_idx[-1])],
        "metrics": m,
        "arima_better_pct": {k: pct_better(m["ARIMA"][k], m["LSTM"][k]) for k in ("rmse", "mae", "mape")},
        "periods_rmse_last_168h": {
            "window": [str(test_idx[w][0]), str(test_idx[w][-1])],
            "ARIMA": period_rmse(test_idx[w], test[w], arima[w]),
            "LSTM": period_rmse(test_idx[w], test[w], lstm[w]),
        },
        "periods_rmse_full_test": {"ARIMA": period_rmse(test_idx, test, arima),
                                   "LSTM": period_rmse(test_idx, test, lstm)},
        "arima_forecast_range": [float(arima.min()), float(arima.max())],
        "actual_range": [float(test.min()), float(test.max())],
    }
    return out


def like_for_like(y, index, origins, forecasts: dict, dm_pairs) -> dict:
    """Score forecasts made from the same origins. forecasts maps name -> (n_origins, H) array."""
    actual = target_matrix(y, origins)
    per_h = {}
    for name, f in forecasts.items():
        rows = [regression_metrics(actual[:, h], f[:, h]) for h in range(actual.shape[1])]
        per_h[name] = {k: [r[k] for r in rows] for k in ("rmse", "mae", "mape", "mean_error", "sd", "band95")}
    at_h = {f"h{h}": {name: {k: v[h - 1] for k, v in per_h[name].items()} for name in forecasts}
            for h in REPORT_HORIZONS}
    bands = {name: band_means(per_h[name]["mae"]) for name in forecasts}
    first, last = next(iter(HORIZON_BANDS)), list(HORIZON_BANDS)[-1]
    target_idx = index[origins + 1]
    out = {
        "n_origins": int(len(origins)),
        "first_origin": str(index[origins[0]]), "last_origin": str(index[origins[-1]]),
        "per_horizon": per_h,
        "at_horizons": at_h,
        "horizon_bands_mae": bands,
        "degradation_pct": {name: degradation_pct(b[first], b[last]) for name, b in bands.items()},
        "periods_rmse_h1": {name: period_rmse(target_idx, actual[:, 0], f[:, 0]) for name, f in forecasts.items()},
        "diebold_mariano": {},
    }
    for a, b in dm_pairs:
        for h in REPORT_HORIZONS:
            out["diebold_mariano"][f"{a} vs {b}, h={h}"] = diebold_mariano(
                actual[:, h - 1] - forecasts[a][:, h - 1], actual[:, h - 1] - forecasts[b][:, h - 1],
                h=h, lag=max(h - 1, 24))
    return out


def monthly_refit_forecasts(y, index, n_train, origins, base_info, order, seasonal_order, log):
    """ARIMA forecasts with parameters re-estimated at the first hour of each month in the test window."""
    f = np.empty((len(origins), MAX_HORIZON))
    starts = [n_train - 1] + [int(t) for t in origins if index[t + 1].day == 1 and index[t + 1].hour == 0]
    refits = []
    for i, t_start in enumerate(starts):
        t_end = starts[i + 1] if i + 1 < len(starts) else origins[-1] + 1
        sel = (origins >= t_start) & (origins < t_end)
        if i == 0:
            params = np.array(list(base_info["params"].values()))
        else:
            res, info = fit_sarima(y[: t_start + 1], order, seasonal_order)
            params = res.params
            refits.append({"fitted_through": str(index[t_start]), "fit_seconds": info["fit_seconds"],
                           "converged": info["converged"]})
            log(f"  refit through {index[t_start]} ({info['fit_seconds']} s)")
        f[sel] = sarima_rolling_forecasts(y, params, origins[sel], order, seasonal_order)
    return f, refits


def run(args) -> dict:
    log = print
    data_path = Path(args.data_dir) / DATA_FILE
    series = load_series(data_path)
    y = series.to_numpy()
    index = series.index
    n = len(y)
    n_train, n_test = chronological_split(n)
    nl = len(drop_leap_days(series))
    results = {
        "script": "arima-vs-lstm/evaluate_onepager_claims.py",
        "data": {"file": DATA_FILE, "sha256": file_sha256(data_path), "rows": n,
                 "first": str(index[0]), "last": str(index[-1])},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "cpu_count": os.cpu_count(), "gpu": "none (CPU run)"},
        "settings": {"split_ratio": SPLIT_RATIO, "lookback": LOOKBACK, "max_horizon": MAX_HORIZON,
                     "sarima": [list(SARIMA_ORDER), list(SARIMA_SEASONAL_ORDER)],
                     "lstm": {"units": LSTM_UNITS, "epochs": LSTM_EPOCHS, "batch": LSTM_BATCH,
                              "validation_split": LSTM_VAL_SPLIT, "patience": LSTM_PATIENCE},
                     "seeds": args.seeds, "periods_hours_half_open": PERIODS,
                     "error_definition": "actual - predicted; band95 = 1.96 x SD (ddof=1)"},
        "claims": CLAIMS,
        "counts": {
            "claimed": CLAIMS["counts"],
            "data": {"total": n, "train": n_train, "test": n_test},
            "data_without_29_feb_2020": dict(zip(("total", "train", "test"),
                                                 (nl, *chronological_split(nl)))),
            "test_window": [str(index[n_train]), str(index[-1])],
        },
    }

    import statsmodels
    results["environment"]["statsmodels"] = statsmodels.__version__

    log(f"Data: {n} rows, train {n_train}, test {n_test}")
    log("Fitting SARIMA(1,1,1)x(1,1,1,24) on the training split ...")
    arima_res, arima_info = fit_sarima(y[:n_train])
    results["arima"] = arima_info
    log(f"  {arima_info['fit_seconds']} s, AIC {arima_info['aic']:.1f}")

    origins = rolling_origins(n_train, n, step=args.origin_step)
    lit_origins = np.arange(n_train - 1, n - 1)  # one-step predictions for every test hour

    log("Rolling-origin SARIMA forecasts (fixed parameters, state updated hourly) ...")
    arima_params = np.asarray(arima_res.params)
    f_arima = sarima_rolling_forecasts(y, arima_params, origins)
    results["arima"]["state_space_check_max_abs_gap"] = check_state_space_forecasts(
        y, arima_params, origins, f_arima, SARIMA_ORDER, SARIMA_SEASONAL_ORDER)

    f_refit, refits = (None, [])
    if not args.no_refit:
        log("Rolling-origin SARIMA with monthly re-estimation ...")
        f_refit, refits = monthly_refit_forecasts(y, index, n_train, origins, arima_info, SARIMA_ORDER,
                                                  SARIMA_SEASONAL_ORDER, log)
    results["arima"]["monthly_refits"] = refits

    search = None
    if args.search_hours > 0:
        log(f"Capped auto_arima search on the last {args.search_hours} training hours ...")
        try:
            search = capped_arima_search(y[:n_train], args.search_hours)
            log(f"  picked {search['order']}x{search['seasonal_order']} in {search['search_seconds']} s")
        except Exception as exc:  # pmdarima missing or search failed
            search = {"error": f"{type(exc).__name__}: {exc}"}
    results["capped_search"] = search
    f_search = None
    search_res = None
    if search and "order" in search and (tuple(search["order"]), tuple(search["seasonal_order"])) != (
            SARIMA_ORDER, SARIMA_SEASONAL_ORDER):
        log("  fitting the searched order on the full training split ...")
        search_res, sinfo = fit_sarima(y[:n_train], search["order"], search["seasonal_order"])
        search["full_train_fit"] = sinfo
        f_search = sarima_rolling_forecasts(y, np.asarray(search_res.params), origins, search["order"],
                                            search["seasonal_order"])

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(y[:n_train].reshape(-1, 1))
    scaled = scaler.transform(y.reshape(-1, 1)).ravel()

    def inv(a):
        return scaler.inverse_transform(np.asarray(a).reshape(-1, 1)).reshape(np.shape(a))

    per_seed = {}
    for seed in args.seeds:
        log(f"Training the LSTM (seed {seed}) ...")
        model, info = train_lstm(scaled[:n_train], seed)
        log(f"  {info['epochs_run']} epochs, best epoch {info['best_epoch']}, {info['train_seconds']} s")
        lit = inv(lstm_recursive_forecasts(model, scaled, lit_origins, max_h=1))[:, 0]
        roll = inv(lstm_recursive_forecasts(model, scaled, origins))
        per_seed[seed] = {"info": info, "literal": lit, "rolling": roll}
    results["lstm_runs"] = [per_seed[s]["info"] for s in args.seeds]
    import tensorflow as tf
    results["environment"]["tensorflow"] = tf.__version__
    try:
        import pmdarima
        results["environment"]["pmdarima"] = pmdarima.__version__
    except ImportError:
        pass

    main_seed = args.seeds[0]
    naive = {"Persistence": persistence_forecasts(y, origins), "Seasonal naive (24 h)":
             seasonal_naive_forecasts(y, origins)}

    results["literal"] = literal_protocol(y, index, n_train, arima_res, per_seed[main_seed]["literal"])
    results["literal"]["seed"] = main_seed
    results["literal"]["lstm_other_seeds"] = {
        str(s): regression_metrics(y[n_train:], per_seed[s]["literal"]) for s in args.seeds}
    if search_res is not None:
        results["literal"]["searched_order_arima"] = regression_metrics(
            y[n_train:], np.asarray(search_res.forecast(steps=n_test)))

    if args.leap_day_variant:
        log("Literal protocol without 29 Feb 2020 (the one-pager's sample counts) ...")
        s2 = drop_leap_days(series)
        y2 = s2.to_numpy()
        n2_train, _ = chronological_split(len(y2))
        res2, info2 = fit_sarima(y2[:n2_train])
        sc2 = MinMaxScaler(feature_range=(0, 1)).fit(y2[:n2_train].reshape(-1, 1))
        z2 = sc2.transform(y2.reshape(-1, 1)).ravel()
        model2, linfo2 = train_lstm(z2[:n2_train], main_seed)
        lit2 = sc2.inverse_transform(
            lstm_recursive_forecasts(model2, z2, np.arange(n2_train - 1, len(y2) - 1), max_h=1)).ravel()
        r2 = literal_protocol(y2, s2.index, n2_train, res2, lit2)
        results["literal_without_29_feb"] = {
            "counts": results["counts"]["data_without_29_feb_2020"], "test_window": r2["test_window"],
            "metrics": r2["metrics"], "arima_better_pct": r2["arima_better_pct"],
            "arima_fit": info2, "lstm": linfo2}

    fc = {"ARIMA": f_arima, "LSTM": per_seed[main_seed]["rolling"]}
    fc["Ensemble 60/40"] = ENSEMBLE_WEIGHTS[0] * fc["ARIMA"] + ENSEMBLE_WEIGHTS[1] * fc["LSTM"]
    if f_refit is not None:
        fc["ARIMA (monthly refit)"] = f_refit
    if f_search is not None:
        fc["ARIMA (searched order)"] = f_search
    fc.update(naive)
    results["like_for_like"] = like_for_like(
        y, index, origins, fc, dm_pairs=[("ARIMA", "LSTM"), ("ARIMA", "Seasonal naive (24 h)"),
                                         ("LSTM", "Persistence")])
    results["like_for_like"]["seed"] = main_seed
    results["like_for_like"]["origin_step_hours"] = args.origin_step
    actual = target_matrix(y, origins)
    results["like_for_like"]["lstm_other_seeds"] = {
        str(s): {f"h{h}": regression_metrics(actual[:, h - 1], per_seed[s]["rolling"][:, h - 1])
                 for h in REPORT_HORIZONS} for s in args.seeds}
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt(v, nd=1):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:,.{nd}f}"


def markdown_report(r: dict) -> str:
    c, lit, lfl = r["claims"], r["literal"], r["like_for_like"]
    L = []
    add = L.append
    add("# One-pager claims: re-run results")
    add("")
    add(f"Generated by `{r['script']}` on `{r['data']['file']}` ({r['data']['rows']:,} rows, "
        f"{r['data']['first']} to {r['data']['last']}; sha256 `{r['data']['sha256'][:12]}...`). CPU only; "
        f"LSTM seed {lit['seed']} unless noted. Error = actual - predicted.")
    add("")
    add("## Sample counts")
    add("")
    add("| | Claimed | In the data | Data without 29 Feb 2020 |")
    add("|---|---|---|---|")
    for k in ("total", "train", "test"):
        add(f"| {k} | {c['counts'][k]:,} | {r['counts']['data'][k]:,} | {r['counts']['data_without_29_feb_2020'][k]:,} |")
    add("")
    add(f"Test window: {r['counts']['test_window'][0]} to {r['counts']['test_window'][1]}.")
    add("")
    add("## Literal protocol (as in the notebook)")
    add("")
    add("SARIMA(1,1,1)x(1,1,1,24) forecasts all test hours in one go from the end of training; the LSTM "
        "predicts each test hour from the actual previous 60 hours.")
    add("")
    add("| Metric | Claimed ARIMA | Claimed LSTM | ARIMA | LSTM | Ensemble 60/40 |")
    add("|---|---|---|---|---|---|")
    ch = c["headline"]
    for k, lab, nd in (("rmse", "RMSE (MW)", 1), ("mae", "MAE (MW)", 1), ("mape", "MAPE (%)", 2)):
        add(f"| {lab} | {ch['ARIMA'][k]} | {ch['LSTM'][k]} | {fmt(lit['metrics']['ARIMA'][k], nd)} | "
            f"{fmt(lit['metrics']['LSTM'][k], nd)} | {fmt(lit['metrics']['Ensemble 60/40'][k], nd)} |")
    add("")
    add("ARIMA better than LSTM by (claimed / measured, % of LSTM): " + ", ".join(
        f"{k.upper()} {ch['arima_better_pct'][k]} / {fmt(lit['arima_better_pct'][k])}" for k in ("rmse", "mae", "mape")))
    add("")
    if "literal_without_29_feb" in r:
        w = r["literal_without_29_feb"]
        add(f"Without 29 Feb 2020 ({w['counts']['train']:,} / {w['counts']['test']:,} hours, test "
            f"{w['test_window'][0]} to {w['test_window'][1]}): ARIMA RMSE {fmt(w['metrics']['ARIMA']['rmse'])}, "
            f"LSTM RMSE {fmt(w['metrics']['LSTM']['rmse'])} MW.")
        add("")
    add("Error distribution (claimed / measured):")
    add("")
    add("| | Mean error ARIMA | Mean error LSTM | SD ARIMA | SD LSTM | ±1.96 SD ARIMA | ±1.96 SD LSTM |")
    add("|---|---|---|---|---|---|---|")
    ce = c["error_distribution"]
    add(f"| Claimed | {ce['ARIMA']['mean_error']} | {ce['LSTM']['mean_error']} | {ce['ARIMA']['sd']} | "
        f"{ce['LSTM']['sd']} | {ce['ARIMA']['band95']} | {ce['LSTM']['band95']} |")
    ma, ml = lit["metrics"]["ARIMA"], lit["metrics"]["LSTM"]
    add(f"| Measured | {fmt(ma['mean_error'])} | {fmt(ml['mean_error'])} | {fmt(ma['sd'])} | {fmt(ml['sd'])} | "
        f"{fmt(ma['band95'])} | {fmt(ml['band95'])} |")
    add("")
    pw = lit["periods_rmse_last_168h"]
    add(f"RMSE by time of day, last 168 test hours ({pw['window'][0]} to {pw['window'][1]}), MW:")
    add("")
    add("| Period | Claimed ARIMA | Claimed LSTM | ARIMA | LSTM | ARIMA, full test | LSTM, full test |")
    add("|---|---|---|---|---|---|---|")
    for p in PERIODS:
        add(f"| {p} | {c['periods_rmse'][p]['ARIMA']} | {c['periods_rmse'][p]['LSTM']} | "
            f"{fmt(pw['ARIMA'][p]['rmse'])} | {fmt(pw['LSTM'][p]['rmse'])} | "
            f"{fmt(lit['periods_rmse_full_test']['ARIMA'][p]['rmse'])} | "
            f"{fmt(lit['periods_rmse_full_test']['LSTM'][p]['rmse'])} |")
    add("")
    add("## Like-for-like rolling-origin evaluation")
    add("")
    add(f"{lfl['n_origins']:,} origins, every {lfl['origin_step_hours']} h from {lfl['first_origin']} to "
        f"{lfl['last_origin']}; each model forecasts 1-24 h ahead using data up to the origin only.")
    add("")
    names = list(lfl["per_horizon"])
    add("| Model | " + " | ".join(f"RMSE h={h}" for h in REPORT_HORIZONS) + " | "
        + " | ".join(f"MAE h={h}" for h in REPORT_HORIZONS) + " | " + " | ".join(f"MAPE h={h}" for h in REPORT_HORIZONS) + " |")
    add("|---|" + "---|" * (3 * len(REPORT_HORIZONS)))
    for nm in names:
        row = [fmt(lfl["at_horizons"][f"h{h}"][nm]["rmse"]) for h in REPORT_HORIZONS]
        row += [fmt(lfl["at_horizons"][f"h{h}"][nm]["mae"]) for h in REPORT_HORIZONS]
        row += [fmt(lfl["at_horizons"][f"h{h}"][nm]["mape"], 2) for h in REPORT_HORIZONS]
        add(f"| {nm} | " + " | ".join(row) + " |")
    add("")
    add("MAE by horizon band (claimed ARIMA / LSTM vs measured), MW:")
    add("")
    add("| Band | Claimed ARIMA | Claimed LSTM | ARIMA | LSTM | ARIMA better by, % of ARIMA (claimed / measured) |")
    add("|---|---|---|---|---|---|")
    for b in HORIZON_BANDS:
        a_, l_ = lfl["horizon_bands_mae"]["ARIMA"][b], lfl["horizon_bands_mae"]["LSTM"][b]
        add(f"| {b} | {c['horizon_bands_mae'][b]['ARIMA']} | {c['horizon_bands_mae'][b]['LSTM']} | "
            f"{fmt(a_)} | {fmt(l_)} | {c['horizon_bands_mae'][b]['arima_better_pct']} / "
            f"{fmt(pct_better(a_, l_, base='a'))} |")
    add("")
    add("Error growth from the 1-6 h band to the 19-24 h band (claimed / measured): " + ", ".join(
        f"{nm} {c['degradation_pct'][nm]}% / {fmt(lfl['degradation_pct'][nm])}%" for nm in ("ARIMA", "LSTM")))
    add("")
    add("Diebold-Mariano tests (squared error; negative = first model better):")
    add("")
    add("| Pair | DM stat | p-value |")
    add("|---|---|---|")
    for k, v in lfl["diebold_mariano"].items():
        add(f"| {k} | {fmt(v['stat'])} | {v['p_value']:.2g} |")
    add("")
    add("1-hour-ahead RMSE by time of day (whole test window), MW:")
    add("")
    add("| Period | " + " | ".join(names) + " |")
    add("|---|" + "---|" * len(names))
    for p in PERIODS:
        add(f"| {p} | " + " | ".join(fmt(lfl["periods_rmse_h1"][nm][p]["rmse"]) for nm in names) + " |")
    add("")
    add("## Run details")
    add("")
    a = r["arima"]
    add(f"- SARIMA fit: {a['fit_seconds']} s on {a['n_train']:,} hours, AIC {fmt(a['aic'])}, converged "
        f"{a['converged']}; state-space forecasts match statsmodels' forecast() to {a['state_space_check_max_abs_gap']:.1e} MW.")
    for run_ in r["lstm_runs"]:
        add(f"- LSTM seed {run_['seed']}: {run_['epochs_run']} epochs (best {run_['best_epoch']}), "
            f"{run_['train_seconds']} s on CPU.")
    s = r.get("capped_search")
    if s and "order" in s:
        add(f"- Capped auto_arima search (notebook settings, last {s['hours']:,} training hours): "
            f"{tuple(s['order'])}x{tuple(s['seasonal_order'])} in {s['search_seconds']} s.")
    elif s:
        add(f"- Capped auto_arima search failed: {s.get('error')}")
    if "searched_order_arima" in lit:
        add(f"- Literal protocol with the searched order: ARIMA RMSE {fmt(lit['searched_order_arima']['rmse'])} MW.")
    add("")
    if len(r["lstm_runs"]) > 1:
        add("LSTM RMSE by training seed (MW):")
        add("")
        add("| Seed | Literal (1 h, actual inputs) | " + " | ".join(
            f"Rolling h={h}" for h in REPORT_HORIZONS) + " |")
        add("|---|---|" + "---|" * len(REPORT_HORIZONS))
        for sd, v in lit["lstm_other_seeds"].items():
            add(f"| {sd} | {fmt(v['rmse'])} | " + " | ".join(
                fmt(lfl["lstm_other_seeds"][sd][f"h{h}"]["rmse"]) for h in REPORT_HORIZONS) + " |")
        add("")
    return "\n".join(L)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", str(REPO_ROOT / "data")),
                   help="folder with final_4year.csv (default: $DATA_DIR or ./data)")
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    p.add_argument("--seeds", default="42", help="comma-separated LSTM seeds; the first is reported")
    p.add_argument("--origin-step", type=int, default=1, help="hours between rolling origins")
    p.add_argument("--search-hours", type=int, default=2016,
                   help="run the notebook's auto_arima search on this many final training hours (0 = skip)")
    p.add_argument("--no-refit", action="store_true", help="skip the monthly ARIMA re-estimation variant")
    p.add_argument("--leap-day-variant", action="store_true",
                   help="also run the literal protocol without 29 Feb 2020 (35,040 rows)")
    a = p.parse_args(argv)
    a.seeds = [int(s) for s in str(a.seeds).split(",") if s.strip()]
    return a


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    results = run(args)
    results["total_seconds"] = round(time.time() - t0, 1)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "onepager_claims.json").write_text(to_json(results), encoding="utf-8")
    (out / "onepager_claims.md").write_text(markdown_report(results), encoding="utf-8")
    print(f"Wrote {out / 'onepager_claims.json'} and {out / 'onepager_claims.md'} "
          f"({results['total_seconds']} s)")


if __name__ == "__main__":
    main()
