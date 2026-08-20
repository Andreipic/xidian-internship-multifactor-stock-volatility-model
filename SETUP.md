# Setup

## Requirements

- Python 3.10 or 3.11 (TensorFlow 2.15/2.16 does not support 3.12 on all
  platforms -- if `pip install -r requirements.txt` fails on TensorFlow,
  check your Python version first with `python3 --version`).
- ~2 GB free disk space (dataset + committed model + generated figures).
- No GPU required. The scripts run on CPU; if TensorFlow prints
  `Could not find cuda drivers on your machine, GPU will not be used`,
  that is expected and harmless -- it is not an error.
- Internet access is only needed if you want to re-fetch live data or
  retrain from scratch (see "Offline vs. live mode" below). Reproducing
  the report's results from the committed files works fully offline.

## Installing

```bash
git clone https://github.com/Andreipic/BYD_volatility_test.git
cd BYD_volatility_test

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

If `pip install` fails on `pykalman`, `PyWavelets`, or `akshare`, install
build tools first (`sudo apt install build-essential` on
Ubuntu/Debian/WSL) and retry.

## Running

Scripts live in `scripts/` and must be run **in order** from inside that
folder:

```bash
cd scripts
python 01_data_collection.py       # Phase 1 + 2: dataset, EDA, external factors
python 02_model_and_prediction.py  # Phase 3: denoising, ARIMA, LSTM, SHAP, intervals
python 03_robustness_and_risk.py   # Phase 4: regimes, stress tests, risk alerts
```

Each script auto-detects the project root (it looks for `data/` and
`models/` one level up from wherever it is launched), so it also works if
you run it from the repo root instead of `scripts/`:

```bash
python scripts/01_data_collection.py
```

`03_robustness_and_risk.py` requires `02_model_and_prediction.py` to have
been run at least once (it loads the trained model and scalers that 02
produces). Running 03 before 02 will fail loudly with a clear
`FileNotFoundError`, rather than silently training a different model.

## Offline vs. live mode

The repo comes with the exact data, trained model, and scalers used to
produce the report, so by default nothing is downloaded or retrained:

| File already present | Script behaviour |
|---|---|
| `data/byd_all_factors_dataset.csv` | `01_` loads it directly, skips all live data collection |
| `models/lstm_recursive_corrected.keras`, `models/scaler_X.pkl`, `models/scaler_y.pkl` | `02_` loads them directly, skips training |

If any of these files are missing, the corresponding script falls back to
live collection / training instead of failing:

- `01_data_collection.py` re-fetches BYD OHLCV and the ten external
  factors from `akshare`, `yfinance`, and FRED. This requires internet
  access and the extra packages at the bottom of `requirements.txt`
  (`yfinance`, `akshare`, `pandas-datareader`, `requests`, `openpyxl`).
  Results will differ from the report, since this pulls more recent data
  than 2015-01-01 to 2025-12-31.
- `02_model_and_prediction.py` retrains the LSTM from scratch. This is
  slower (CPU-only, no GPU needed but expect several minutes) and, due to
  training stochasticity, will not reproduce the report's exact MAPE
  values, though it should land in a similar range if everything else is
  unchanged.

Each script prints a warning to the console (and to its log file, see
below) when it falls back to live mode, so it is always clear which mode
produced a given run.

## Outputs

- `figures/`: all report figures, overwritten on every run.
- `logs/<script_name>.log`: full console output of the last run of each
  script, overwritten on every run. Check here first if a number looks
  off or a figure did not get generated -- the log will show exactly
  which code path (offline/committed-file vs. live-fetch/retrain) was
  taken.
- `data/phase4_robustness_table.csv`: machine-readable version of the
  Phase 4 robustness summary (Table 9 in the report), written by
  `03_robustness_and_risk.py`.
- `models/metrics_summary.csv`: machine-readable version of the Phase 3
  model comparison (Table 3 in the report), written by
  `02_model_and_prediction.py`.

## Troubleshooting

- **`UnboundLocalError` involving `mdates` or another module inside a
  function**: check that no `import` statement was accidentally added
  inside `main()` after the module-level import at the top of the file --
  a local import shadows the global one for the entire function body,
  even before the line where it appears.
- **A stress-scenario or MC-Dropout figure looks visually similar to
  another figure in the same run**: this is expected at a glance (same
  axes, same price range, same interval-band style) and is not a bug --
  compare the underlying MAPE/alert-rate numbers in the console output
  rather than the figures alone if you need to confirm they used
  different data.
- **Figures don't update after editing a script**: `matplotlib.use("Agg")`
  is set at the top of each script (no GUI backend), so figures never
  pop up on screen -- they are written straight to `figures/`. Check the
  file's modified timestamp (`ls -la figures/`) if unsure whether a
  script actually re-ran.
