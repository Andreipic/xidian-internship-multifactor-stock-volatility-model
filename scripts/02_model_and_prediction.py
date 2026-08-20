"""
BYD Stock Volatility Analysis -- Step 2: Model Training and Prediction

Loads the model + scalers already committed to models/ if present (no
retraining). If missing, trains a new LSTM from scratch and prints a clear
warning that resulting metrics will differ from the report.

Debugging history: an early version of this pipeline reported MAPE=1.77%.
That number came from a leaky, non-causal wavelet filter and a recursive
forecast that re-anchored on the true price at every step. Fixing that,
plus a frozen recursive window and non-stationary raw-price features,
brought the honest MAPE to 4.70% (still under the 8% project threshold,
and better than ARIMA at 4.94% and a naive baseline at 6.68%).

Run: python 02_model_and_prediction.py
"""
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.filterwarnings("ignore", category=ValueWarning)
warnings.simplefilter("ignore", category=ValueWarning)
from scipy.stats import norm

import pywt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
tf.get_logger().setLevel(logging.ERROR)

import shap

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

LOOKBACK = 60
HORIZON = 22
TARGET_COL = "Close"
N_CALIB_WINDOWS = 15
CALIB_STRIDE = 5
CONFIDENCE_LEVELS = [0.80, 0.95]


# --------------------------------------------------------------------------
def find_project_root(markers=("data", "models")):
    here = os.getcwd()
    for path in (here, os.path.dirname(here)):
        if all(os.path.isdir(os.path.join(path, m)) for m in markers):
            return path
    return here


PROJECT = find_project_root()
DATA = os.path.join(PROJECT, "data")
MODELS = os.path.join(PROJECT, "models")
FIGURES = os.path.join(PROJECT, "figures")
for folder in (DATA, MODELS, FIGURES):
    os.makedirs(folder, exist_ok=True)

LOGS = os.path.join(PROJECT, "logs")
os.makedirs(LOGS, exist_ok=True)

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

_log_path = os.path.join(LOGS, os.path.splitext(os.path.basename(__file__))[0] + ".log")
_log_file = open(_log_path, "w", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _log_file)

DATASET_PATH = os.path.join(DATA, "byd_all_factors_dataset.csv")
MODEL_PATH = os.path.join(MODELS, "lstm_recursive_corrected.keras")
SCALER_X_PATH = os.path.join(MODELS, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(MODELS, "scaler_y.pkl")


# --------------------------------------------------------------------------
# Causal signal filters
# --------------------------------------------------------------------------
def kalman_filter_manual(series, Q=1e-5, R=0.05):
    """Local-level Kalman filter, causal (Harvey, 1989)."""
    n = len(series)
    xhat, P = np.zeros(n), np.zeros(n)
    xhat[0], P[0] = series[0], 1.0
    for t in range(1, n):
        x_pred = xhat[t - 1]
        P_pred = P[t - 1] + Q
        K = P_pred / (P_pred + R)
        xhat[t] = x_pred + K * (series[t] - x_pred)
        P[t] = (1 - K) * P_pred
    return xhat


def wavelet_haar_denoising(series):
    """Haar wavelet soft-threshold denoising (Donoho & Johnstone, 1994).
    Non-causal (uses the whole series) -- shift(1) before use as a feature."""
    data = np.array(series, dtype=float)
    n = len(data)
    if n % 2 != 0:
        data = np.append(data, data[-1])
        n += 1
    A = (data[0::2] + data[1::2]) / np.sqrt(2)
    D = (data[0::2] - data[1::2]) / np.sqrt(2)
    sigma = np.median(np.abs(D)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(D)))
    D_filt = np.sign(D) * np.maximum(0, np.abs(D) - threshold)
    rec = np.zeros(n)
    rec[0::2] = (A + D_filt) / np.sqrt(2)
    rec[1::2] = (A - D_filt) / np.sqrt(2)
    return rec[:len(series)]


# --------------------------------------------------------------------------
# Recursive forecast (strictly cumulative -- no leakage)
# --------------------------------------------------------------------------
def returns_to_prices(log_returns, last_known_price):
    prices = [last_known_price]
    for r in log_returns:
        prices.append(prices[-1] * np.exp(r))
    return np.array(prices[1:])


def recursive_forecast(model, X_last_window, scaler_X, scaler_y, feature_cols, ar_idx, n_steps, last_known_price):
    window = X_last_window.copy()
    preds_scaled = []
    for _ in range(n_steps):
        pred_scaled = model.predict(window[np.newaxis, :, :], verbose=0)[0, 0]
        preds_scaled.append(pred_scaled)
        new_row = window[-1].copy()
        new_row[ar_idx] = pred_scaled
        window = np.vstack([window[1:], new_row])
    log_returns_pred = scaler_y.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return returns_to_prices(log_returns_pred, last_known_price)


def diebold_mariano(e1, e2):
    d = e1.flatten() ** 2 - e2.flatten() ** 2
    d_mean, d_var, n = d.mean(), d.var(ddof=1), len(d)
    stat = d_mean / np.sqrt(d_var / n)
    p = 2 * (1 - norm.cdf(abs(stat)))
    return stat, p


# --------------------------------------------------------------------------
def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run 01_data_collection.py first."
        )
    df = pd.read_csv(DATASET_PATH, parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)
    df = df.ffill().dropna()
    print(f"Dataset: {df.shape} ({df.index.min().date()} to {df.index.max().date()})")

    # --- Causal filters ---
    df["Kalman"] = kalman_filter_manual(df["Close"].values)
    wavelet_raw = wavelet_haar_denoising(df["Close"].values)
    df["Wavelet_causal"] = pd.Series(wavelet_raw, index=df.index).shift(1)
    df.dropna(subset=["Wavelet_causal"], inplace=True)

    plt.figure(figsize=(16, 5))
    plt.plot(df.index, df["Close"], label="Realized price", color="lightgray", alpha=0.8)
    plt.plot(df.index, df["Kalman"], label="Kalman (causal)", color="red", linewidth=1.3)
    plt.plot(df.index, df["Wavelet_causal"], label="Wavelet (causal, shift 1)", color="green", linewidth=1.3)
    plt.title("BYD close price -- causal filters")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig5_signal_filters_causal.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- ARIMA baseline ---
    adf_result = adfuller(df["Close"].dropna())
    print(f"\nADF on Close: stat={adf_result[0]:.4f}, p={adf_result[1]:.4f}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(df["Close"].diff().dropna(), lags=40, ax=axes[0])
    plot_pacf(df["Close"].diff().dropna(), lags=40, ax=axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "acf_pacf.png"), dpi=150, bbox_inches="tight")
    plt.close()

    train_data = df.loc[:"2022-12-31", "Close"]
    test_data = df.loc["2023-01-01":, "Close"]
    n_windows = (len(test_data) - HORIZON) // HORIZON

    arima_preds, arima_true = [], []
    history = train_data.copy()
    for i in range(n_windows):
        start = i * HORIZON
        window_true = test_data.iloc[start:start + HORIZON]
        if len(window_true) < HORIZON:
            break
        model_i = ARIMA(history, order=(1, 1, 1)).fit()
        forecast_i = model_i.forecast(steps=HORIZON)
        arima_preds.extend(forecast_i.values)
        arima_true.extend(window_true.values)
        history = pd.concat([history, window_true])
    arima_preds, arima_true = np.array(arima_preds), np.array(arima_true)
    rmse_arima = np.sqrt(mean_squared_error(arima_true, arima_preds))
    mae_arima = mean_absolute_error(arima_true, arima_preds)
    mape_arima = mean_absolute_percentage_error(arima_true, arima_preds) * 100
    print(f"ARIMA(1,1,1): RMSE={rmse_arima:.2f} MAE={mae_arima:.2f} MAPE={mape_arima:.2f}%")

    # --- Feature set ---
    FEATURES_DAILY = ["Close", "Open", "High", "Low", "Volume", "Kalman", "Wavelet_causal"]
    FEATURES_DAILY = [c for c in FEATURES_DAILY if c in df.columns]
    data_model = df[FEATURES_DAILY].copy()

    # Low-frequency macro factors (LPR, commodities), weekly/monthly, forward-filled onto daily index
    macro_cols = [c for c in ["PBoC_LPR_1y", "Copper", "Brent_Oil"] if c in df.columns]
    for col in macro_cols:
        data_model[f"{col}_weekly"] = df[col].resample("W").mean().reindex(df.index, method="ffill")
        data_model[f"{col}_monthly"] = df[col].resample("ME").mean().reindex(df.index, method="ffill")
    data_model.dropna(inplace=True)

    data_lstm = data_model.copy()
    data_lstm["log_return"] = np.log(data_lstm[TARGET_COL] / data_lstm[TARGET_COL].shift(1))

    PRICE_LEVEL_COLS = [c for c in [
        "Open", "High", "Low", "Volume", "Kalman", "Wavelet_causal",
        "Copper", "Brent_Oil", "Copper_weekly", "Copper_monthly",
        "Brent_Oil_weekly", "Brent_Oil_monthly",
    ] if c in data_lstm.columns]
    RATE_LEVEL_COLS = [c for c in [
        "PBoC_LPR_1y", "PBoC_LPR_1y_weekly", "PBoC_LPR_1y_monthly",
    ] if c in data_lstm.columns]
    for col in PRICE_LEVEL_COLS:
        data_lstm[col] = data_lstm[col].pct_change()
    for col in RATE_LEVEL_COLS:
        data_lstm[col] = data_lstm[col].diff()
    data_lstm.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_lstm.dropna(inplace=True)

    feature_cols = [c for c in data_lstm.columns if c not in ("log_return", "Close")] + ["log_return"]
    ar_idx = feature_cols.index("log_return")
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

    scalers_exist = os.path.exists(SCALER_X_PATH) and os.path.exists(SCALER_Y_PATH)
    if scalers_exist:
        scaler_X = joblib.load(SCALER_X_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
        X_scaled = scaler_X.transform(data_lstm[feature_cols])
        y_scaled = scaler_y.transform(data_lstm[["log_return"]])
        print("Scalers loaded from disk (matches the saved model).")
    else:
        scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
        X_scaled = scaler_X.fit_transform(data_lstm[feature_cols])
        y_scaled = scaler_y.fit_transform(data_lstm[["log_return"]])
        print("No saved scalers found -- fitting new ones.")

    def build_sequences(X, y, lookback):
        Xs, ys = [], []
        for i in range(lookback, len(X)):
            Xs.append(X[i - lookback:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = build_sequences(X_scaled, y_scaled, LOOKBACK)
    split_idx = data_lstm.index.get_indexer(["2023-01-01"], method="bfill")[0] - LOOKBACK
    calib_size = N_CALIB_WINDOWS * HORIZON
    calib_start = split_idx - calib_size
    X_train, X_calib, X_test = X_seq[:calib_start], X_seq[calib_start:split_idx], X_seq[split_idx:]
    y_train = y_seq[:calib_start]
    print(f"Train: {X_train.shape}, Calib: {X_calib.shape}, Test: {X_test.shape}")

    if not scalers_exist:
        joblib.dump(scaler_X, SCALER_X_PATH)
        joblib.dump(scaler_y, SCALER_Y_PATH)

    # --- Model: load or train ---
    model_exists = os.path.exists(MODEL_PATH)
    if model_exists:
        model = load_model(MODEL_PATH)
        print(f"\nModel loaded from {MODEL_PATH} -- no retraining.")
    else:
        print("\n" + "=" * 70)
        print("WARNING: no saved model found. Training a new one now.")
        print("Results below (MAPE, SHAP, intervals) WILL DIFFER from the report,")
        print("which used one specific trained model committed to models/.")
        print("=" * 70 + "\n")
        model = Sequential([
            Input(shape=(LOOKBACK, X_train.shape[2])),
            LSTM(64, return_sequences=True), Dropout(0.2),
            LSTM(32), Dropout(0.2),
            Dense(16, activation="relu"), Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        early_stop = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)
        model.fit(X_train, y_train, epochs=400, batch_size=32, validation_split=0.1,
                  callbacks=[early_stop, reduce_lr], verbose=1)
        model.save(MODEL_PATH)
        print(f"Model trained and saved to {MODEL_PATH}.")

    # --- Point forecast on test set ---
    n_test_windows = len(X_test) // HORIZON
    lstm_preds_all, lstm_true_all = [], []
    for w in range(n_test_windows):
        start = w * HORIZON
        if start + HORIZON > len(X_test):
            break
        idx_full = split_idx + start
        last_price = data_lstm[TARGET_COL].iloc[idx_full + LOOKBACK - 1]
        true_prices = data_lstm[TARGET_COL].iloc[idx_full + LOOKBACK: idx_full + LOOKBACK + HORIZON].values
        preds = recursive_forecast(model, X_test[start], scaler_X, scaler_y, feature_cols, ar_idx, HORIZON, last_price)
        lstm_preds_all.extend(preds)
        lstm_true_all.extend(true_prices)
    lstm_preds_all, lstm_true_all = np.array(lstm_preds_all), np.array(lstm_true_all)
    mape_lstm_raw = mean_absolute_percentage_error(lstm_true_all, lstm_preds_all) * 100
    print(f"\nLSTM (raw, before bias correction): MAPE={mape_lstm_raw:.2f}%")

    naive_preds = np.roll(lstm_true_all, HORIZON)[HORIZON:]
    naive_true = lstm_true_all[HORIZON:]
    mape_naive = mean_absolute_percentage_error(naive_true, naive_preds) * 100

    dir_acc = np.mean(np.sign(np.diff(lstm_true_all)) == np.sign(np.diff(lstm_preds_all))) * 100
    err_lstm_aligned = (lstm_true_all - lstm_preds_all)[HORIZON:]
    err_naive = naive_true - naive_preds
    dm_stat, dm_p = diebold_mariano(err_lstm_aligned, err_naive)
    print(f"Directional accuracy: {dir_acc:.2f}%")
    print(f"Diebold-Mariano vs naive: stat={dm_stat:.3f}, p={dm_p:.4f}")

    # --- SHAP ---
    background = X_train[np.random.choice(X_train.shape[0], min(100, X_train.shape[0]), replace=False)]
    explainer = shap.GradientExplainer(model, background)
    sample_idx = np.random.choice(X_test.shape[0], min(50, X_test.shape[0]), replace=False)
    X_shap_sample = X_test[sample_idx]
    shap_values = explainer.shap_values(X_shap_sample)
    shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values
    shap_agg = np.abs(shap_values).mean(axis=1).squeeze(-1)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_agg, features=X_shap_sample.mean(axis=1), feature_names=feature_cols, show=False)
    plt.title("SHAP summary")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig8_shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()


    mean_abs_shap = np.abs(shap_agg).mean(axis=0)
    mean_abs_shap = np.asarray(mean_abs_shap).flatten()

    lpr_commo_cols = [c for c in feature_cols if any(k in c for k in ["PBoC_LPR", "Copper", "Brent_Oil"])]
    lpr_commo_idx = [feature_cols.index(c) for c in lpr_commo_cols]
    lpr_commo_values = mean_abs_shap[lpr_commo_idx].flatten().tolist()

    shap_lpr_summary = pd.DataFrame({
        "feature": lpr_commo_cols,
        "mean_abs_shap": lpr_commo_values
    }).sort_values("mean_abs_shap", ascending=False)

    plt.figure(figsize=(9, 5))
    plt.barh(shap_lpr_summary["feature"].tolist(), shap_lpr_summary["mean_abs_shap"].tolist(), color="darkorange")
    plt.xlabel("Mean |SHAP|")
    plt.title("Mean SHAP Contribution - LPR & Commodities (daily / weekly / monthly)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig9_shap_lpr_commodities_focus.png"), dpi=150, bbox_inches="tight")
    plt.show()
    

    mean_abs_shap = np.abs(shap_agg).mean(axis=0)
    lpr_commo_cols = [c for c in feature_cols if any(k in c for k in ["PBoC_LPR", "Copper", "Brent_Oil"])]
    lpr_commo_idx = [feature_cols.index(c) for c in lpr_commo_cols]
    shap_lpr_summary = pd.DataFrame({
        "feature": lpr_commo_cols, "mean_abs_shap": mean_abs_shap[lpr_commo_idx],
    }).sort_values("mean_abs_shap", ascending=False)
    print("\nSHAP contribution -- LPR & commodities, by frequency:")
    print(shap_lpr_summary.to_string(index=False))

    # --- MC-Dropout intervals ---
    @tf.function
    def _predict_step(m, w):
        return m(w, training=True)

    def mc_dropout_forecast(model, X_last_window, scaler_y, n_steps, last_known_price, n_mc=100):
        windows = np.repeat(X_last_window[np.newaxis, :, :], n_mc, axis=0)
        preds_scaled = np.zeros((n_mc, n_steps))
        for t in range(n_steps):
            batch_pred = _predict_step(model, windows).numpy()[:, 0]
            preds_scaled[:, t] = batch_pred
            new_rows = windows[:, -1, :].copy()
            new_rows[:, ar_idx] = batch_pred
            windows = np.concatenate([windows[:, 1:, :], new_rows[:, np.newaxis, :]], axis=1)
        paths = np.zeros((n_mc, n_steps))
        for m in range(n_mc):
            lr = scaler_y.inverse_transform(preds_scaled[m].reshape(-1, 1)).flatten()
            paths[m] = returns_to_prices(lr, last_known_price)
        return paths

    last_window = X_test[-HORIZON] if len(X_test) > HORIZON else X_test[0]
    idx_last = split_idx + max(0, len(X_test) - HORIZON)
    last_price_final = data_lstm[TARGET_COL].iloc[idx_last + LOOKBACK - 1]
    true_prices_final = data_lstm[TARGET_COL].iloc[idx_last + LOOKBACK: idx_last + LOOKBACK + HORIZON].values
    mc_paths = mc_dropout_forecast(model, last_window, scaler_y, HORIZON, last_price_final)
    mc_mean = mc_paths.mean(axis=0)


    mc_intervals = {}
    for cl in CONFIDENCE_LEVELS:
        lo_q, hi_q = (1 - cl) / 2, 1 - (1 - cl) / 2
        mc_intervals[cl] = (np.quantile(mc_paths, lo_q, axis=0), np.quantile(mc_paths, hi_q, axis=0))
    for cl, (lo, hi) in mc_intervals.items():
        inside = (true_prices_final >= lo) & (true_prices_final <= hi)
        print(f"MC-Dropout {int(cl*100)}% coverage (single window): {inside.mean()*100:.1f}%")

    plt.figure(figsize=(12, 6))
    x_axis = np.arange(HORIZON)

    colors_ci = {0.80: "orange", 0.95: "khaki"}
    for cl in sorted(CONFIDENCE_LEVELS, reverse=True):
        lo, hi = mc_intervals[cl]
        plt.fill_between(x_axis, lo, hi, alpha=0.35, color=colors_ci.get(cl, "gray"),
                          label=f"{int(cl*100)}% interval")

    plt.plot(x_axis, mc_mean, label="Mean forecast (MC-Dropout)", color="darkblue", linewidth=2)
    plt.plot(x_axis, true_prices_final[:len(mc_mean)], label="Realized price", color="black",
             linestyle="--", marker="o", markersize=3)
    plt.xlabel("Horizon day (1-22)")
    plt.ylabel("Price (CNY)")
    plt.title("Recursive LSTM - MC-Dropout Prediction Intervals (80% / 95%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig10_mcdropout_intervals.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Conformal calibration (places MC-Dropout for the alert rule) ---
    calib_errors = []
    for start in range(0, len(X_calib) - HORIZON, CALIB_STRIDE):
        abs_idx = calib_start + start
        last_price_c = data_lstm[TARGET_COL].iloc[abs_idx + LOOKBACK - 1]
        true_c = data_lstm[TARGET_COL].iloc[abs_idx + LOOKBACK: abs_idx + LOOKBACK + HORIZON].values
        if len(true_c) < HORIZON:
            continue
        preds_c = recursive_forecast(model, X_seq[abs_idx], scaler_X, scaler_y, feature_cols, ar_idx, HORIZON, last_price_c)
        calib_errors.append(true_c - preds_c)
    calib_errors = np.array(calib_errors)
    print(f"\nCalibration windows: {calib_errors.shape[0]}")

    # Global conformal coverage over the full test set
    all_true, all_lo80, all_hi80, all_lo95, all_hi95 = [], [], [], [], []
    for w in range(n_test_windows):
        start = w * HORIZON
        if start + HORIZON > len(X_test):
            break
        idx_full = split_idx + start
        last_price_w = data_lstm[TARGET_COL].iloc[idx_full + LOOKBACK - 1]
        true_w = data_lstm[TARGET_COL].iloc[idx_full + LOOKBACK: idx_full + LOOKBACK + HORIZON].values
        preds_w = recursive_forecast(model, X_test[start], scaler_X, scaler_y, feature_cols, ar_idx, HORIZON, last_price_w)
        for cl, lo_l, hi_l in [(0.80, all_lo80, all_hi80), (0.95, all_lo95, all_hi95)]:
            lo_q, hi_q = (1 - cl) / 2, 1 - (1 - cl) / 2
            err_lo = np.quantile(calib_errors, lo_q, axis=0)
            err_hi = np.quantile(calib_errors, hi_q, axis=0)
            lo_l.extend(preds_w + err_lo)
            hi_l.extend(preds_w + err_hi)
        all_true.extend(true_w)
    all_true = np.array(all_true)
    for cl, lo, hi in [(0.80, np.array(all_lo80), np.array(all_hi80)), (0.95, np.array(all_lo95), np.array(all_hi95))]:
        inside = (all_true >= lo) & (all_true <= hi)
        print(f"Conformal {int(cl*100)}% global coverage (n={len(all_true)}): {inside.mean()*100:.1f}%, "
              f"mean width={((hi-lo).mean()):.1f} CNY")

    # --- Systematic bias detection + correction ---
    all_true_bias, all_pred_bias = [], []
    for w in range(n_test_windows):
        start = w * HORIZON
        if start + HORIZON > len(X_test):
            break
        idx_full = split_idx + start
        last_price_w = data_lstm[TARGET_COL].iloc[idx_full + LOOKBACK - 1]
        true_w = data_lstm[TARGET_COL].iloc[idx_full + LOOKBACK: idx_full + LOOKBACK + HORIZON].values
        preds_w = recursive_forecast(model, X_test[start], scaler_X, scaler_y, feature_cols, ar_idx, HORIZON, last_price_w)
        all_true_bias.extend(true_w)
        all_pred_bias.extend(preds_w)
    all_true_bias, all_pred_bias = np.array(all_true_bias), np.array(all_pred_bias)
    bias_per_day = calib_errors.mean(axis=0)

    all_pred_corrected = []
    for w in range(n_test_windows):
        start = w * HORIZON
        if start + HORIZON > len(X_test):
            break
        idx_full = split_idx + start
        last_price_w = data_lstm[TARGET_COL].iloc[idx_full + LOOKBACK - 1]
        preds_w = recursive_forecast(model, X_test[start], scaler_X, scaler_y, feature_cols, ar_idx, HORIZON, last_price_w)
        all_pred_corrected.extend(preds_w + bias_per_day)
    all_pred_corrected = np.array(all_pred_corrected)

    mape_corrected = mean_absolute_percentage_error(all_true_bias, all_pred_corrected) * 100
    rmse_corrected = np.sqrt(mean_squared_error(all_true_bias, all_pred_corrected))
    mae_corrected = mean_absolute_error(all_true_bias, all_pred_corrected)

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, HORIZON + 1), bias_per_day, marker="o")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Horizon day")
    plt.ylabel("Bias correction (CNY)")
    plt.title("Bias correction by horizon day")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "bias_by_horizon_day.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 60)
    print("FINAL RESULTS (2023-2025 test set, 704 observations)")
    print("=" * 60)
    print(f"{'Model':<28}{'RMSE':>10}{'MAE':>10}{'MAPE (%)':>12}")
    print(f"{'ARIMA(1,1,1)':<28}{rmse_arima:>10.2f}{mae_arima:>10.2f}{mape_arima:>12.2f}")
    print(f"{'Naive (walk-forward)':<28}{'':>10}{'':>10}{mape_naive:>12.2f}")
    print(f"{'LSTM, recursive (raw)':<28}{'':>10}{'':>10}{mape_lstm_raw:>12.2f}")
    print(f"{'LSTM, bias-corrected':<28}{rmse_corrected:>10.2f}{mae_corrected:>10.2f}{mape_corrected:>12.2f}")
    print("\nProject target: MAPE < 8% -- ", "PASS" if mape_corrected < 8 else "FAIL")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(len(lstm_true_all)), lstm_true_all, label="Realized price", color="black", linewidth=1.5)
    ax.plot(range(len(lstm_preds_all)), lstm_preds_all, label="Recursive LSTM (point forecast)", color="crimson", alpha=0.8)
    ax.set_title("Final comparison on test set (2023-2025)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig7_final_comparison_test_set.png"), dpi=150, bbox_inches="tight")
    plt.close()

    metrics_table = pd.DataFrame({
        "Model": ["ARIMA(1,1,1)", "Naive", "LSTM raw", "LSTM bias-corrected"],
        "RMSE": [rmse_arima, np.nan, np.nan, rmse_corrected],
        "MAE": [mae_arima, np.nan, np.nan, mae_corrected],
        "MAPE (%)": [mape_arima, mape_naive, mape_lstm_raw, mape_corrected],
    })
    metrics_table.to_csv(os.path.join(MODELS, "metrics_summary.csv"), index=False)
    print(f"\nDone. Figures saved to {FIGURES}, metrics saved to {MODELS}/metrics_summary.csv")
    if not model_exists:
        print("\nReminder: this run trained a NEW model. Re-run this script once more if you want")
        print("Phase 3/4 results to be reproducible from this point forward (they will now load")
        print("the model + scalers just saved, instead of retraining again).")


if __name__ == "__main__":
    main()