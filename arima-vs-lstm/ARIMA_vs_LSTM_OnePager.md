# Comparative Analysis: ARIMA vs LSTM for Time Series Forecasting
## Statistical Machine Learning versus Deep Learning in Electricity Demand Prediction

> **Re-run with `evaluate_onepager_claims.py` (September 2026).**
> - The first draft of this page reported figures that no notebook produced, and they did not hold up. Each number below comes from that script's CPU run on `final_4year.csv`. The run uses LSTM seed 42 unless noted, with five seeds for the spread.
> - Where a claim failed, the measured result replaces it.
> - `results/onepager_claims.md` shows the draft's claims next to the measured values; `results/onepager_claims.json` has everything else.

### Executive Summary
This study compares a traditional statistical method (seasonal ARIMA) with a modern deep learning approach (LSTM) for hourly electricity demand forecasting. It uses four years of real-world hourly demand data for Delhi (2020-2023).

The answer depends on how the two models are compared:
- **Under the original notebook's protocol, the LSTM wins by a wide margin.** There, ARIMA forecasts the whole 146-day test window in one go, while the LSTM predicts each hour from the actual previous hours.
- **When both forecast from the same points in time with the same information, ARIMA is ahead** at 1 and 6 hours, and level with the LSTM at 24 hours.

### Research Objective
To empirically evaluate whether advanced deep learning architectures provide superior predictive performance compared to classical statistical models in time series forecasting, specifically in the context of electricity demand prediction with strong seasonal patterns.

### Experimental Design
- **Dataset**: 35,064 hourly observations of electricity demand (MW) for Delhi. This is 1,461 days, with 29 February 2020 included. The draft's 35,040 / 31,536 / 3,504 are the counts you get with that day removed. Removing it changes little: ARIMA RMSE 880.0 MW, LSTM 111.6 MW under the notebook's protocol.
- **Temporal Coverage**: January 2020 - December 2023 (4 complete years), no missing hours
- **Target Variable**: Hourly Demand Met (Megawatts)
- **Data Split**: chronological 90/10.
  - Training: 31,557 hours, up to 2023-08-07 20:00.
  - Testing: 3,507 hours, 2023-08-07 21:00 to 2023-12-31 23:00.
- **Methodology**: Univariate time series analysis, so both models get the same inputs. There are two evaluation protocols:
  1. **Notebook protocol**, as in `ARIMA_vs_LSTM_Comparison.ipynb`. ARIMA forecasts all 3,507 test hours from the end of training. The LSTM predicts each test hour from the actual previous 60 hours.
  2. **Like-for-like protocol.** From each of 3,484 hourly origins in the test window, both models forecast 1-24 hours ahead using only data up to that origin.
     - ARIMA keeps its training-set parameters, and its state is updated with each observed hour.
     - The LSTM gets the actual previous 60 hours and feeds its own predictions back for later hours.

---

## Model Specifications

| **ARIMA (Statistical Model)** | **LSTM (Deep Learning Model)** |
|:-------------------------------|:--------------------------------|
| **Model Class**: Seasonal AutoRegressive Integrated Moving Average | **Model Class**: Recurrent Neural Network with LSTM cells |
| **Configuration**: SARIMA(1,1,1)×(1,1,1)₂₄ (statsmodels SARIMAX) | **Architecture**: Input(60,1) → LSTM(50) → Dense(1), 10,451 parameters |
| **Seasonal Component**: 24-hour periodicity | **Temporal Window**: 60-hour lookback period |
| **Optimization Method**: Maximum likelihood. The order is the notebook's fallback; the notebook's AIC search, capped to the last 2,016 training hours, picks (0,1,2)×(1,0,0)₂₄, which forecasts worse (see below) | **Optimization Method**: Adam optimizer (MSE loss), up to 20 epochs, early stopping on the last 10% of training windows |
| **Model Interpretability**: High - decomposable components | **Model Interpretability**: Low - non-linear transformations |
| **Computational Requirements**: CPU only, 54 s to fit on 31,557 hours | **Computational Requirements**: CPU only (no GPU used), 243-322 s to train per seed |

---

## Performance Evaluation

### Quantitative Results on Hold-out Test Set (3,507 observations, notebook protocol)

| **Performance Metric** | **ARIMA** | **LSTM** | **Relative Performance** |
|:----------------------|:----------|:---------|:-------------------------|
| Root Mean Squared Error (RMSE) | 860.2 MW | 105.9 MW | LSTM superior by 87.7% |
| Mean Absolute Error (MAE) | 695.1 MW | 77.0 MW | LSTM superior by 88.9% |
| Mean Absolute Percentage Error (MAPE) | 20.26% | 2.00% | LSTM superior by 90.1% |

The draft reported ARIMA ahead by 4.5% (RMSE 341.28 vs 357.42 MW). That did not reproduce. The notebook protocol is not a like-for-like test:
- ARIMA's forecast for the last test hour is made 3,507 hours in advance.
- The LSTM always sees the true demand up to the previous hour.

Across five training seeds the LSTM's RMSE here ranges from 94.9 to 106.9 MW.

### Like-for-like Results (same 3,484 origins, same information)

| **Forecast Horizon** | **ARIMA RMSE** | **LSTM RMSE** | **Relative Performance** | **Diebold-Mariano p-value** |
|:---------------------|:---------------|:--------------|:-------------------------|:----------------------------|
| 1 hour ahead | 65.7 MW | 105.9 MW | ARIMA superior by 38.0% | < 0.001 |
| 6 hours ahead | 246.8 MW | 288.9 MW | ARIMA superior by 14.6% | < 0.001 |
| 24 hours ahead | 371.6 MW | 391.4 MW | ARIMA superior by 5.1% | 0.39 (not significant) |

**Statistical Significance**: This is a Diebold-Mariano test on squared errors, with a HAC variance using 24 lags. ARIMA's advantage is significant at 1 and 6 hours but not at 24 hours. For reference, repeating the same hour of the previous day gives 333.6 MW at 1 hour and 333.9 MW at 24 hours. At 24 hours that beats both models (ARIMA vs this seasonal naive forecast: p < 0.001).

Across the five LSTM seeds the like-for-like RMSE ranges:
- 95.0-107.0 MW at 1 hour;
- 256.9-342.6 MW at 6 hours;
- 374.6-657.0 MW at 24 hours.

ARIMA is ahead of every seed at every one of these horizons.

---

## Temporal Performance Analysis

### Prediction Accuracy Pattern (168-hour sample window)

The window is the last 168 test hours, 25-31 December 2023, as plotted in the notebook. Hours are taken from the start of each range, so 10:00-14:00 means the hours stamped 10 to 13.

| **Time Period** | **ARIMA RMSE** | **LSTM RMSE** | **1-hour-ahead ARIMA / LSTM, whole test window** | **Observation** |
|:----------------|:---------------|:--------------|:------------------------------------------------|:----------------|
| Peak Hours (10:00-14:00) | 2,080.8 MW | 111.6 MW | 61.3 / 134.3 MW | ARIMA's one-shot forecast, made 3,340+ hours earlier, has lost most of the daily cycle and badly under-forecasts the daytime peak |
| Evening Peak (18:00-22:00) | 1,245.7 MW | 103.9 MW | 69.3 / 124.4 MW | Transitions are hard: of these three periods, ARIMA's 1-hour error is highest in the evening ramp, the LSTM's at midday |
| Off-Peak (00:00-06:00) | 467.0 MW | 37.8 MW | 39.9 / 45.9 MW | Lowest error rates during stable demand periods |

### Error Distribution Analysis
The error is actual minus predicted, so a positive mean means under-forecasting. The ±1.96 SD band is the interval that would hold 95% of errors if they were normal.

| **Statistical Measure** | **ARIMA (notebook protocol)** | **LSTM (notebook protocol)** | **ARIMA (1 h ahead, like-for-like)** | **LSTM (1 h ahead, like-for-like)** |
|:------------------------|:----------|:----------|:----------|:----------|
| Mean Prediction Error | +297 MW | -7 MW | 0 MW | -8 MW |
| Standard Deviation | 807 MW | 106 MW | 66 MW | 106 MW |
| ±1.96 SD band | ±1,583 MW | ±207 MW | ±129 MW | ±207 MW |

The narrower band belongs to the LSTM under the notebook protocol and to ARIMA when both forecast one hour ahead from the same data.

---

## Forward-Looking Forecast Performance

### 24-Hour Ahead Predictive Capability Assessment
This uses the like-for-like protocol; MAE is averaged over the horizons in each band and all 3,484 origins. The notebook's own 24-hour forecast starts after the last hour of data, so it cannot be scored.

| **Forecast Horizon** | **ARIMA MAE** | **LSTM MAE** | **Performance Delta** |
|:---------------------|:--------------|:-------------|:---------------------|
| 1-6 hours | 120.3 MW | 164.8 MW | ARIMA +37.0% accuracy |
| 7-12 hours | 213.4 MW | 229.0 MW | ARIMA +7.3% accuracy |
| 13-18 hours | 247.5 MW | 265.7 MW | ARIMA +7.3% accuracy |
| 19-24 hours | 252.5 MW | 274.4 MW | ARIMA +8.7% accuracy |

**Forecast Degradation Rate**: ARIMA's error grows by 109.9% from the first band to the last, and the LSTM's by 66.5%. ARIMA starts much lower and stays ahead in every band.

Repeating the previous day's value for the same hour gives an MAE of 223.0-223.7 MW in every band. By MAE, that simple forecast beats the LSTM from 9 hours ahead and ARIMA from 11 hours ahead.

---

## Comparative Analysis: Model Strengths and Applications

*General guidance. Only the points marked "measured" were tested in this study.*

### ARIMA - Optimal Use Cases
| **Criterion** | **Rationale** |
|:--------------|:--------------|
| **Seasonal Time Series** | Handles regular cyclical patterns explicitly. Measured: with hourly state updates it was the most accurate model here at 1-6 hours |
| **Limited Training Data** | Few parameters to estimate (5 here) |
| **Interpretability Requirements** | Decomposable into trend, seasonal, and residual components |
| **Computational Constraints** | CPU-based execution. Measured: one fit took 54 s. The notebook's automatic order search on the full training set grew past 19 GB of RAM before finishing a model, so the search has to run on a sample |
| **Real-time Deployment** | Low-latency predictions, no specialized hardware needed |

### LSTM - Optimal Use Cases
| **Criterion** | **Rationale** |
|:--------------|:--------------|
| **Non-linear Dynamics** | Captures complex, non-stationary patterns |
| **Multivariate Inputs** | Incorporates external variables (weather, economic indicators); not used here |
| **Long-range Dependencies** | Can learn dependencies within its input window (60 hours here) |
| **Large-scale Data** | Performance tends to scale with dataset size. Measured: with 31,497 training windows, results varied noticeably between training seeds |
| **Feature Learning** | Automatic feature extraction from raw inputs |

---

## Conclusions and Recommendations

### Primary Finding
**Which model wins depends on the evaluation protocol:**
- **Notebook protocol:** the LSTM is far ahead, with RMSE 105.9 vs 860.2 MW.
- **Same origins and information:** ARIMA is ahead, with 38.0% lower RMSE at 1 hour and 14.6% at 6 hours (both significant). At 24 hours ARIMA is 5.1% lower, which is not significant.

The draft's headline of ARIMA with 4.5% lower RMSE was not reproduced under either protocol. It still holds that deep learning does not automatically outperform a classical model on this data.

### Why the Results Look Like This

1. **Domain Characteristics Alignment**: The inherent 24-hour periodicity in electricity demand suits ARIMA's seasonal terms. Once its state is updated with each observed hour, ARIMA's 1-hour RMSE (65.7 MW) is far below persistence (235.3 MW).

2. **Sample Efficiency**: With 31,557 training hours, the SARIMA model estimates 5 parameters and the LSTM 10,451. The LSTM's results depend on the training seed: its 24-hour RMSE ranges from 374.6 to 657.0 MW across five seeds.

3. **Univariate Constraints**: The restriction to single-variable analysis negates LSTM's primary advantage of multi-dimensional feature processing.

4. **Numerical Stability**: Neither model is stable far beyond the data it has seen:
   - ARIMA's 3,507-hour forecast drifts and loses the daily cycle.
   - The LSTM's recursive forecasts vary with the seed.
   - A different ARIMA order changes results a lot. The (0,1,2)×(1,0,0)₂₄ chosen by the capped search scores 2,441.8 MW RMSE under the notebook protocol and 76.5 MW at 1 hour ahead.

### Strategic Recommendations

| **Recommendation** | **Implementation Strategy** | **Measured Outcome** |
|:-------------------|:---------------------------|:--------------------|
| **Immediate Deployment** | Implement SARIMA(1,1,1)×(1,1,1)₂₄ with its state updated every hour, for short horizons | 38.0% lower 1-hour RMSE than the LSTM (65.7 vs 105.9 MW) |
| **Ensemble Approach** | Weighted average: 60% ARIMA, 40% LSTM | Better than either model alone at 6 and 24 hours (RMSE 232.2 and 355.8 MW), slightly behind ARIMA at 1 hour (70.5 MW); still behind the seasonal naive forecast at 24 hours (333.9 MW) |
| **Feature Engineering** | Incorporate temperature, holidays, economic indicators | Not quantified in this study |
| **Continuous Learning** | Update the ARIMA state with each new hour; re-estimating the parameters monthly changed RMSE by less than 0.3 MW | The hourly state updates, not re-estimation, make the difference |
| **Baseline Check** | Compare every model with the same hour of the previous day | By MAE, that baseline beats ARIMA from 11 hours ahead and the LSTM from 9 |

### Theoretical Implications
This empirical study is consistent with the **No Free Lunch theorem** in machine learning: model selection must be driven by data characteristics rather than algorithmic complexity. It also shows that the evaluation protocol can matter as much as the model class. For time series with strong seasonal patterns and limited exogenous variables, classical statistical methods remain highly competitive.

### Future Research Directions
- Investigation of hybrid models combining ARIMA's seasonal decomposition with LSTM's pattern recognition
- Evaluation of transformer architectures for long-term electricity demand forecasting
- Multi-horizon forecasting performance comparison across different time scales, and models that use weather forecasts as inputs

---

**Citation**: *Comparative Analysis of ARIMA and LSTM Models for Electricity Demand Forecasting, Delhi Metropolitan Grid Data (2020-2023)*

**Reproducibility Note**: Every number on this page is produced by `python arima-vs-lstm/evaluate_onepager_claims.py --seeds 42,43,44,45,46 --leap-day-variant` with `DATA_DIR` pointing at the data folder. That run took 37 minutes on a 24-core CPU without a GPU. The settings, the draft's claims, per-horizon curves and all baselines are in `results/onepager_claims.json`, and the side-by-side tables are in `results/onepager_claims.md`.
