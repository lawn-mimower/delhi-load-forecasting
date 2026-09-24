"""Offline tests for the helpers in arima-vs-lstm/evaluate_onepager_claims.py.

No data, TensorFlow or GPU needed; the state-space check fits nothing and runs a
Kalman filter on 300 synthetic points.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "arima-vs-lstm" / "evaluate_onepager_claims.py"


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("evaluate_onepager_claims", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hourly(start, end):
    return pd.date_range(start, end, freq="h")


def test_split_sizes_in_data_and_in_one_pager(ev):
    assert ev.chronological_split(35064) == (31557, 3507)
    assert ev.chronological_split(35040) == (31536, 3504)


def test_one_pager_counts_equal_data_without_leap_day(ev):
    idx = hourly("2020-01-01 00:00", "2023-12-31 23:00")
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    assert len(s) == 35064
    assert len(ev.drop_leap_days(s)) == 35040


def test_regression_metrics(ev):
    m = ev.regression_metrics([100, 200, 300, 400], [110, 190, 330, 400])
    assert m["n"] == 4
    assert m["rmse"] == pytest.approx(np.sqrt(275))
    assert m["mae"] == pytest.approx(12.5)
    assert m["mape"] == pytest.approx(6.25)
    assert m["mean_error"] == pytest.approx(-7.5)  # actual - predicted
    assert m["sd"] == pytest.approx(np.sqrt(875 / 3))
    assert m["band95"] == pytest.approx(1.96 * np.sqrt(875 / 3))


def test_regression_metrics_skips_non_finite(ev):
    m = ev.regression_metrics([100, 200, np.nan], [110, np.inf, 5])
    assert m["n"] == 1 and m["rmse"] == pytest.approx(10)


def test_percentages_follow_the_one_pagers_arithmetic(ev):
    # Headline: (LSTM - ARIMA) / LSTM.
    assert ev.pct_better(341.28, 357.42) == pytest.approx(4.5, abs=0.05)
    assert ev.pct_better(268.15, 279.83) == pytest.approx(4.2, abs=0.05)
    assert ev.pct_better(4.82, 5.13) == pytest.approx(6.0, abs=0.05)
    # Horizon table: (LSTM - ARIMA) / ARIMA.
    assert ev.pct_better(198.3, 212.7, base="a") == pytest.approx(7.3, abs=0.05)
    assert ev.pct_better(312.8, 327.5, base="a") == pytest.approx(4.7, abs=0.05)
    # Degradation: last band over first band.
    assert ev.degradation_pct(198.3, 312.8) == pytest.approx(57.7, abs=0.05)
    assert ev.degradation_pct(212.7, 327.5) == pytest.approx(54.0, abs=0.05)


def test_hour_bands_are_half_open(ev):
    idx = hourly("2023-01-01 00:00", "2023-01-02 23:00")
    assert ev.hour_mask(idx, 10, 14).sum() == 8  # hours 10-13 on two days
    y = np.zeros(len(idx))
    pred = np.where(np.asarray(idx.hour) == 14, 100.0, 0.0)  # error only at 14:00
    out = ev.period_rmse(idx, y, pred)
    assert out["Peak Hours (10:00-14:00)"] == {"n": 8, "rmse": 0.0}
    assert out["Off-Peak (00:00-06:00)"]["n"] == 12


def test_rolling_origins_and_targets(ev):
    y = np.arange(100, dtype=float)
    origins = ev.rolling_origins(n_train=90, n=100, max_h=5)
    assert origins.tolist() == [89, 90, 91, 92, 93, 94]
    t = ev.target_matrix(y, origins, max_h=5)
    assert t[0].tolist() == [90, 91, 92, 93, 94]  # first origin forecasts the first test hours
    assert t[-1].tolist() == [95, 96, 97, 98, 99]
    assert ev.rolling_origins(90, 100, max_h=5, step=2).tolist() == [89, 91, 93]


def test_naive_forecasts(ev):
    y = np.arange(100, dtype=float)
    origins = np.array([50, 60])
    assert ev.persistence_forecasts(y, origins, max_h=3).tolist() == [[50] * 3, [60] * 3]
    sn = ev.seasonal_naive_forecasts(y, origins, max_h=24, season=24)
    assert sn[0, 0] == 27 and sn[0, -1] == 50  # y[t + h - 24]
    with pytest.raises(ValueError):
        ev.seasonal_naive_forecasts(y, origins, max_h=25, season=24)


def test_band_means(ev):
    per_h = np.arange(1, 25, dtype=float)
    assert ev.band_means(per_h) == {"1-6 hours": 3.5, "7-12 hours": 9.5, "13-18 hours": 15.5,
                                    "19-24 hours": 21.5}


def test_make_windows_matches_notebook_sequences(ev):
    data = np.arange(70, dtype=float)
    X, y = ev.make_windows(data, lookback=60)
    assert X.shape == (10, 60, 1) and y.tolist() == list(range(60, 70))
    assert X[3, :, 0].tolist() == list(range(3, 63))


def test_state_space_forecasts_match_statsmodels(ev):
    sm = pytest.importorskip("statsmodels.tsa.statespace.sarimax")
    rng = np.random.default_rng(0)
    n = 300
    y = 50 + 10 * np.sin(np.arange(n) * 2 * np.pi / 4) + np.cumsum(rng.normal(size=n))
    kw = dict(order=(1, 1, 1), seasonal_order=(1, 0, 0, 4),
              enforce_stationarity=False, enforce_invertibility=False)
    params = np.array([0.3, -0.2, 0.5, 1.0])
    res = sm.SARIMAX(y, **kw).filter(params)
    ssm = res.model.ssm
    origins = np.array([199, 250, 290])
    got = ev.state_space_forecasts(res.filter_results.predicted_state, ssm["design"], ssm["transition"],
                                   origins, 6, ssm["obs_intercept"], ssm["state_intercept"])
    for row, t in zip(got, origins):
        ref = sm.SARIMAX(y[: t + 1], **kw).filter(params).forecast(6)
        np.testing.assert_allclose(row, ref, rtol=0, atol=1e-8)


def test_diebold_mariano_sign_and_degenerate_case(ev):
    pytest.importorskip("scipy")
    rng = np.random.default_rng(1)
    e2 = rng.normal(scale=2.0, size=500)
    e1 = rng.normal(scale=1.0, size=500)
    out = ev.diebold_mariano(e1, e2, h=1)
    assert out["stat"] < 0 and out["p_value"] < 0.01
    same = ev.diebold_mariano(e1, e1, h=1)
    assert np.isnan(same["stat"])


def test_json_writer_uses_fixed_decimals(ev):
    import json

    obj = {"a": 3.5, "b": [1, 2.5, None], "c": {"p": 8.6e-54, "ok": True, "name": "x"}, "d": float("nan")}
    text = ev.to_json(obj)
    assert '"a": 3.5000' in text and "2.5000" in text and "8.6000e-54" in text
    back = json.loads(text)
    assert back["a"] == 3.5 and back["b"] == [1, 2.5, None] and back["c"]["ok"] is True
    assert back["d"] is None
