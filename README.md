# Delhi hourly electricity demand forecasting

Delhi's hourly electricity demand in 2020–2023 ranged from about 1,100 MW (a December early morning) to 7,700 MW (a June afternoon). How much of that can be predicted, and from what? This project began as a Smart India Hackathon 2024 attempt to forecast demand from weather. It then grew into head-to-head comparisons on four years of hourly data: a weather-only random forest, STL decomposition plus an MLP, SARIMA vs LSTM, and a lag-feature LightGBM vs a PyTorch LSTM.

Two findings stand out:
- Recent demand predicts the next hour far better than same-hour weather.
- Several of the comparisons are less fair than they look.

Both are spelled out below.

## Notebooks

**`sih-2024/`**: exploratory hackathon notebooks.
- `Trials/Untitled.ipynb`: 2022 daily peak demand plotted against daily minimum and maximum temperature.
- `Data_Study.ipynb`: loads the 2023 hourly file and plots temperature over the year.
- `XGBonDelhi.ipynb`: plots the 2023 data and fits an XGBoost regressor. The regressor predicts *temperature*, not demand (see issues).
- `Model Comparision.ipynb`: on the four-year file.
  - A random forest from four same-hour weather features (temperature, humidity, solar radiation, wind speed) to demand, on a chronological 80/20 split, with feature importances.
  - An STL decomposition (24 h period) with an MLP fitted to the residual.

**`arima-vs-lstm/`**: demand-only (univariate) comparisons on the four-year file, with a chronological 90/10 split.
- `ARIMA_vs_LSTM_Comparison.ipynb`:
  - A seasonal ARIMA from a capped `pmdarima.auto_arima` search (m = 24), with a SARIMAX fallback, against a Keras LSTM (60-hour window, 50 units).
  - Reports RMSE, MAE and MAPE, then forecasts 24 hours past the end of the data.
- `GPU_Optimized_ARIMA_LSTM.ipynb`: a TensorFlow variant.
  - Fixed SARIMA(1,1,1)(1,1,1,24) on the last 5,000 training hours.
  - A two-layer LSTM with mixed precision on the GPU, falling back to CPU.
- `ARIMA_vs_LSTM_OnePager.md`: a draft write-up of the study. Its numbers are not reproduced (see Results).
- `check_gpu_setup.py`: checks the NVIDIA driver, CUDA, cuDNN and TensorFlow GPU support, and whether CuPy, Numba and PyTorch are installed. If anything is missing it writes, but does not run, a `fix_gpu_setup.sh` suggestion script.

**`model-comparisons/PyTorch_GPU_Reactive_LGBM_vs_LSTM.ipynb`** compares two one-hour-ahead models, both of which see actual past demand:
- LightGBM on calendar features, demand lags (1, 2, 3, 24 and 168 h) and rolling statistics.
- A two-layer PyTorch LSTM on the previous 60 hours.

The `TRAINING_RUN` variable sets the LSTM's training length:

| Value | Max epochs | Early-stopping patience |
|---|---|---|
| `standard` (default) | 30 | 5 |
| `long` | 100 | 100 |

It uses CUDA with mixed precision if available. On CPU the LSTM shrinks to hidden size 32 and batch 64, against 64 and 256 on GPU.

## Data

The data is **not included**. The compiled dataset will be available on Hugging Face: **TODO: add Hugging Face dataset link** (link TBD).

The notebooks read from `DATA_DIR`, which defaults to `../data`: a `data/` folder at the repository root, when each notebook runs from its own folder. `Trials/` uses `../../data`, which is the same folder.

| File | Used by | Contents |
|---|---|---|
| `final_4year.csv` | `Model Comparision`, all ARIMA/LSTM/LightGBM notebooks | 35,064 hourly rows, 2020-01-01 00:00 to 2023-12-31 23:00, no gaps. `Date and Time`, `Day of Week`, `Holiday Status`, `Hourly Demand Met (in MW)`, weather columns, `Time` |
| `final_modified.csv` | `Data_Study`, `XGBonDelhi` | 8,760 hourly rows for 2023: the same weather columns plus `Hour` and `Month` |
| `Delhi_2022_PowerVSTemp.csv` | `Trials/Untitled` | Daily 2022 rows: `Date`, `Peak Demand (in MW)`, `Minimum Temprature (in °C)`, `Maximum Temprature (in °C)` (spelling as in the file) |

The weather columns (`temp`, `feelslike`, `dew`, `humidity`, `precip`, `windspeed`, `solarradiation`, `uvindex`, `conditions`, `icon`, …) have the names of a Visual Crossing weather export. Three of its downloads were in US units, so in 2,856 hours of `final_4year.csv` `temp`, `feelslike` and `dew` are in °F, wind in mph, `precip` in inches and `visibility` in miles: 2020-07-21 to 2020-08-30, 2021-02-11 to 2021-03-20 and 2021-08-31 to 2021-10-09. The notebooks use the columns as they are. The code doesn't record where the demand series came from. The XGBoost notebook originally read the 2023 file from a Kaggle input named `delhi-hourly-2023`.

Don't mix the two files' demand columns. The weather columns in `final_modified.csv` match `final_4year.csv` for 2023 exactly. Its `Hourly Demand Met (in MW)` column, however:
- runs from about 118,000 to 237,000, against about 1,400 to 7,300 in `final_4year.csv`;
- is not a constant multiple of the Delhi series, so it is a different series.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # versions tested (Python 3.12, CPU) are noted in the file

python -m pytest tests                 # 15 offline tests; no data or GPU needed
python arima-vs-lstm/check_gpu_setup.py

cd model-comparisons                   # run notebooks from their own folder
DATA_DIR=/abs/path/to/data TRAINING_RUN=standard \
  jupyter nbconvert --to notebook --execute --output-dir /tmp/runs \
  PyTorch_GPU_Reactive_LGBM_vs_LSTM.ipynb
```

LightGBM and PyTorch share one kernel in that notebook, and both use OpenMP. If the LSTM cell stalls after the LightGBM cell, start Jupyter with `OMP_WAIT_POLICY=PASSIVE`. It wasn't needed in our CPU runs.

CPU run times:
- The LightGBM vs LSTM notebook took 1–2.5 minutes on the full four years.
- The SIH notebooks take under a minute.
- The `auto_arima` search dominates `ARIMA_vs_LSTM_Comparison.ipynb`. On six months of data the notebook took 7.5 minutes, most of it in the first search cell, which hit its 5-minute timeout.

## Results

These results come from CPU runs of the committed notebooks on the full `final_4year.csv`.

**One hour ahead.** The notebook is `PyTorch_GPU_Reactive_LGBM_vs_LSTM.ipynb` with `TRAINING_RUN=standard` and the CPU LSTM settings. The test window is the last 10% of the data: 3,507 hours, 2023-08-07 21:00 to 2023-12-31 23:00. Each prediction uses actual demand up to the previous hour. Repeated runs (seed 42) gave identical numbers.

| Model | RMSE (MW) | MAE (MW) | MAPE |
|---|---|---|---|
| LightGBM (lags + calendar features) | 94.8 | 51.6 | 1.35% |
| PyTorch LSTM (60 h window; early-stopped at epoch 14) | 146.1 | 107.4 | 2.72% |
| *Reference: previous hour's value (computed separately)* | 235.9 | 189.8 | 5.40% |

The notebook's own table labels the LightGBM row "ARIMA". Its LightGBM RMSE is also inflated by 17 back-filled hours; without them it is 69.6 MW.

**Weather only.** This is the random forest in `Model Comparision.ipynb`, with no demand history. On its own 80/20 split (test window 2023-03-14 19:00 to 2023-12-31 23:00):
- It scores RMSE ≈ 742 MW, with temperature by far the most important feature (≈ 0.61).
- On the same window, the previous hour's value alone scores 227 MW.

**ARIMA vs LSTM.** No figures are reported here, because the two models are evaluated differently (see issues). Read `ARIMA_vs_LSTM_OnePager.md` as a draft:
- Its sample counts (35,040 / 31,536 / 3,504) don't match the data (35,064 / 31,557 / 3,507).
- No notebook computes its per-period and per-horizon tables.
- Its headline (ARIMA ahead of LSTM by 4.5% RMSE) has not been reproduced.

## Known issues and limitations

- **`XGBonDelhi.ipynb`**
  - Its target is `temp`, which is also an input feature (target leakage).
  - Its test months (Jan–Feb 2023) come before its training months (Apr–Dec 2023).
  - An earlier plotting cell rescales columns in place (temperature × 100), so its RMSE is in hundredths of a degree.
  - The split plot still marks 2015.
- **STL leakage in `Model Comparision.ipynb`.** STL is fitted on the whole series, test period included, and the "final" prediction adds back the true test-period trend and seasonal parts. That gives RMSE ≈ 125 MW on the full data, but trend plus seasonal alone gives ≈ 126 MW. The MLP adds almost nothing; the score comes from the test period itself.
- **ARIMA vs LSTM is not like for like.** ARIMA forecasts the whole ~146-day test window in one go from the end of training. The LSTM predicts each hour from the actual previous 60 hours.
- **`GPU_Optimized_ARIMA_LSTM.ipynb` needs about four months of data** (roughly 2,900+ hourly rows). With less, its validation set is empty and it stops with `KeyError: 'val_loss'`: 2,800 rows failed and 3,000 ran. Its validation batches are the *earliest* 10% of training windows.
- **LightGBM vs LSTM**
  - LightGBM early-stops on the test set.
  - Its predictions are labelled "ARIMA" in the printed table, the plots and the saved files.
  - It drops the first 168 hours before splitting, so the first 17 test hours are back-filled with a constant.
  - The LSTM's validation windows are also in its training loader.
- **Outputs are written next to the notebooks.**
  - `ARIMA_vs_LSTM_Comparison.ipynb` overwrites the committed `model_parameters.json`, and picks up `arima_search_state.json` left by a previous run.
  - The LightGBM notebook writes `pytorch_config.json`, predictions and weights into its folder.
  - The committed JSON files are outputs of earlier runs, not results. Run on a copy to keep them.
- **General limitations**
  - Models use observed weather, not weather forecasts.
  - Each comparison is one chronological split, with no rolling-origin evaluation and no tuning beyond the notebooks' defaults.

## Repository layout

```
sih-2024/            Smart India Hackathon 2024 notebooks (data study, XGBoost, model comparison, 2022 trial)
arima-vs-lstm/       ARIMA vs LSTM notebooks, one-page write-up, GPU check script, saved run configs
model-comparisons/   LightGBM vs PyTorch LSTM notebook (TRAINING_RUN=standard|long)
tests/               offline tests for check_gpu_setup.py and the TRAINING_RUN option
requirements.txt
```

Licence: MIT — see [LICENSE](LICENSE).
