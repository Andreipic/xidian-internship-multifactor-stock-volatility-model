"""
BYD Stock Volatility Analysis -- Step 3: Robustness and Risk Warning

Loads the exact model + scalers validated in Step 2 (no retraining here --
this script fails loudly if they are missing, rather than silently training
a different model that would give different regime/stress numbers).

Run: python 03_robustness_and_risk.py
(requires 02_model_and_prediction.py to have been run at least once)
"""
import os
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

from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
tf.get_logger().setLevel(logging.ERROR)
import pywt

LOOKBACK = 60
HORIZON = 22
TARGET_COL = "Close"
N_CALIB_WINDOWS = 15
CALIB_STRIDE = 5
ALERT_THRESHOLD = 0.05
ALERT_CONFIDENCE = 0.80
FIXED_ANCHOR_DATES = ["2022-12-12", "2023-08-09", "2024-04-11", "2024-12-06", "2025-08-06"]


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


def kalman_filter_manual(series, Q=1e-5, R=0.05):
    n = len(series)
    x_hat, P = np.zeros(n), np.zeros(n)
    x_hat[0], P[0] = series[0], 1.0
    for t in range(1, n):
        x_pred = x_hat[t - 1]
        P_pred = P[t - 1] + Q
        K = P_pred / (P_pred + R)
        x_hat[t] = x_pred + K * (series[t] - x_pred)
        P[t] = (1 - K) * P_pred
    return x_hat


def wavelet_haar_causal(series, level=4, min_window=64):
    n = len(series)
    out = np.zeros(n)
    out[:min_window] = series[:min_window]
    for t in range(min_window, n):
        window = np.array(series[:t + 1], dtype=np.float64, copy=True)
        coeffs = pywt.wavedec(window, "haar", level=min(level, pywt.dwt_max_level(len(window), "haar")))
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(window)))
        coeffs[1:] = [pywt.threshold(c, uthresh, mode="soft") for c in coeffs[1:]]
        rec = pywt.waverec(coeffs, "haar")
        out[t] = rec[t] if t < len(rec) else rec[-1]
    return out


def returns_to_prices(log_returns, last_known_price):
    prices = [last_known_price]
    for r in log_returns:
        prices.append(prices[-1] * np.exp(r))
    return np.array(prices[1:])


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"{DATASET_PATH} not found. Run 01_data_collection.py first.")
    if not (os.path.exists(SCALER_X_PATH) and os.path.exists(SCALER_Y_PATH)):
        raise FileNotFoundError(
            f"Scalers not found in {MODELS}. Run 02_model_and_prediction.py first "
            "so Phase 4 uses the exact model and scalers validated in Phase 3."
        )
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run 02_model_and_prediction.py first -- Phase 4 "
            "must load the exact validated model, not train its own."
        )

    df = pd.read_csv(DATASET_PATH, parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)
    df = df.ffill().dropna()
    print(f"Dataset: {df.shape} ({df.index.min().date()} to {df.index.max().date()})")

    df["Kalman"] = kalman_filter_manual(df["Close"].values)
    df["Wavelet_causal"] = wavelet_haar_causal(df["Close"].values)

    macro_cols = [c for c in ["PBoC_LPR_1y", "Copper", "Brent_Oil"] if c in df.columns]
    for col in macro_cols:
        df[f"{col}_weekly"] = df[col].resample("W").last().reindex(df.index, method="ffill")
        df[f"{col}_monthly"] = df[col].resample("ME").last().reindex(df.index, method="ffill")

    FEATURES_DAILY = ["Close", "Open", "High", "Low", "Volume", "Kalman", "Wavelet_causal"] + \
        [c for c in ["PBoC_LPR_1y_weekly", "PBoC_LPR_1y_monthly",
                     "Copper_weekly", "Copper_monthly",
                     "Brent_Oil_weekly", "Brent_Oil_monthly"] if c in df.columns]
    FEATURES_DAILY = [c for c in FEATURES_DAILY if c in df.columns]
    data_model = df[FEATURES_DAILY].copy()

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

    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    X_scaled = scaler_X.transform(data_lstm[feature_cols])
    y_scaled = scaler_y.transform(data_lstm[["log_return"]])

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
    X_calib, X_test = X_seq[calib_start:split_idx], X_seq[split_idx:]
    print(f"Calib: {X_calib.shape}, Test: {X_test.shape}")

    model = load_model(MODEL_PATH)
    print(f"Loaded validated model from {MODEL_PATH} (no retraining).")

    def recursive_forecast(model, X_last_window, last_known_price):
        window = X_last_window.copy()
        preds_scaled = []
        for _ in range(HORIZON):
            pred_scaled = model.predict(window[np.newaxis, :, :], verbose=0)[0, 0]
            preds_scaled.append(pred_scaled)
            new_row = window[-1].copy()
            new_row[ar_idx] = pred_scaled
            window = np.vstack([window[1:], new_row])
        log_returns_pred = scaler_y.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        return returns_to_prices(log_returns_pred, last_known_price)

    # --- Bias correction, from the calibration block ---
    calib_errors = []
    for start in range(0, len(X_calib) - HORIZON, CALIB_STRIDE):
        abs_idx = calib_start + start
        last_price_c = data_lstm[TARGET_COL].iloc[abs_idx + LOOKBACK - 1]
        true_c = data_lstm[TARGET_COL].iloc[abs_idx + LOOKBACK: abs_idx + LOOKBACK + HORIZON].values
        if len(true_c) < HORIZON:
            continue
        preds_c = recursive_forecast(model, X_seq[abs_idx], last_price_c)
        calib_errors.append(true_c - preds_c)
    calib_errors = np.array(calib_errors)
    bias_per_day = calib_errors.mean(axis=0)
    print(f"Calibration windows: {calib_errors.shape[0]}")

    def recursive_forecast_corrected(model, X_last_window, last_known_price):
        return recursive_forecast(model, X_last_window, last_known_price) + bias_per_day

    def conformal_interval(point_forecast, confidence_level):
        lo_q, hi_q = (1 - confidence_level) / 2, 1 - (1 - confidence_level) / 2
        err_lo = np.quantile(calib_errors, lo_q, axis=0)
        err_hi = np.quantile(calib_errors, hi_q, axis=0)
        return point_forecast + err_lo, point_forecast + err_hi

    def risk_alert_flags(true_prices, lower, upper, threshold=ALERT_THRESHOLD):
        below = true_prices < lower * (1 - threshold)
        above = true_prices > upper * (1 + threshold)
        return below | above

    # --- Step 1: volatility regimes ---
    test_prices = data_lstm[TARGET_COL].iloc[split_idx + LOOKBACK:]
    rolling_vol = test_prices.pct_change().rolling(20).std() * np.sqrt(252)
    q33, q66 = rolling_vol.quantile(0.33), rolling_vol.quantile(0.66)
    regime = pd.Series("Normal", index=test_prices.index)
    regime[rolling_vol < q33] = "Calm"
    regime[rolling_vol > q66] = "Volatile"
    print("\nRegime distribution:")
    print(regime.value_counts())

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    colors = {"Calm": "green", "Normal": "gray", "Volatile": "red"}
    for r, c in colors.items():
        mask = regime == r
        axes[0].scatter(test_prices.index[mask], test_prices[mask], s=6, color=c, label=r)
    axes[0].legend()
    axes[0].set_title("Market regimes on test set (2023-2025)")
    axes[1].plot(rolling_vol.index, rolling_vol * 100, color="purple", linewidth=0.8)
    axes[1].axhline(q33 * 100, color="green", linestyle="--")
    axes[1].axhline(q66 * 100, color="red", linestyle="--")
    axes[1].set_ylabel("20d realized vol (%, annualized)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig11_regimes_on_test_set.png"), dpi=150, bbox_inches="tight")
    plt.close()

    n_test_windows = len(X_test) // HORIZON
    regime_records = []
    for w in range(n_test_windows):
        start = w * HORIZON
        if start + HORIZON > len(X_test):
            break
        idx_full = split_idx + start
        last_price_w = data_lstm[TARGET_COL].iloc[idx_full + LOOKBACK - 1]
        true_w = data_lstm[TARGET_COL].iloc[idx_full + LOOKBACK: idx_full + LOOKBACK + HORIZON].values
        window_dates = data_lstm.index[idx_full + LOOKBACK: idx_full + LOOKBACK + HORIZON]
        preds_w = recursive_forecast_corrected(model, X_test[start], last_price_w)
        for d, t, p in zip(window_dates, true_w, preds_w):
            regime_records.append({"date": d, "regime": regime.get(d, "Normal"), "true": t, "pred": p})
    df_bt = pd.DataFrame(regime_records)

    results_regime = []
    for r in ["Calm", "Normal", "Volatile"]:
        sub = df_bt[df_bt["regime"] == r]
        if len(sub) == 0:
            continue
        mape_r = mean_absolute_percentage_error(sub["true"], sub["pred"]) * 100
        results_regime.append({"Regime": r, "N_days": len(sub), "LSTM_MAPE": round(mape_r, 2)})
    df_regime = pd.DataFrame(results_regime)
    print("\nMAPE by regime:")
    print(df_regime.to_string(index=False))

    # --- Stress scenario functions ---
    def apply_policy_shock(raw_df, start_day, price_drop=0.08, noise_days=10, noise_mult=2.0,
                            sector_drop=0.05, vix_spike=20):
        d = raw_df.copy()
        shock_date = d.index[start_day]
        for col in ["Close", "Open", "High", "Low"]:
            if col in d.columns:
                d.loc[shock_date:, col] = d.loc[shock_date:, col] * (1 - price_drop)
        noise_end = min(start_day + noise_days, len(d))
        noise_dates = d.index[start_day:noise_end]
        noise = np.random.normal(0, d["Close"].pct_change().std() * noise_mult, len(noise_dates))
        d.loc[noise_dates, "Close"] = d.loc[noise_dates, "Close"] * (1 + noise)
        for col in ["SSE", "Tesla"]:
            if col in d.columns:
                d.loc[shock_date:, col] = d.loc[shock_date:, col] * (1 - sector_drop)
        if "VIX" in d.columns:
            d.loc[shock_date:, "VIX"] = d.loc[shock_date:, "VIX"] + vix_spike
        return d, start_day

    def apply_market_crash(raw_df, start_day, crash_duration=40, total_drop=0.25,
                            recovery_days=20, recovery_pct=0.10, noise_mult=1.5):
        d = raw_df.copy()
        daily_drop = total_drop / crash_duration
        pre_vol = d["Close"].pct_change().std()
        cum_factor = 1.0
        for i in range(start_day, min(start_day + crash_duration, len(d))):
            idx = d.index[i]
            cum_factor *= (1 - daily_drop)
            noise = np.random.normal(0, pre_vol * noise_mult)
            d.loc[idx, "Close"] = raw_df.loc[idx, "Close"] * cum_factor * (1 + noise)
        recovery_start = min(start_day + crash_duration, len(d) - 1)
        daily_recovery = recovery_pct / recovery_days
        recovery_factor = cum_factor
        for i in range(recovery_start, min(recovery_start + recovery_days, len(d))):
            idx = d.index[i]
            recovery_factor *= (1 + daily_recovery)
            d.loc[idx, "Close"] = raw_df.loc[idx, "Close"] * recovery_factor
        tail_start = min(recovery_start + recovery_days, len(d))
        d.iloc[tail_start:] = raw_df.iloc[tail_start:]
        for col in ["Open", "High", "Low"]:
            if col in d.columns:
                d[col] = d[col] * (d["Close"] / raw_df["Close"]).fillna(1.0)
        end_idx = min(recovery_start + recovery_days, len(d) - 1)
        if "SSE" in d.columns:
            d.loc[d.index[start_day]:d.index[end_idx], "SSE"] *= 0.80
        if "VIX" in d.columns:
            d.loc[d.index[start_day]:d.index[end_idx], "VIX"] += 15
        return d, start_day

    def rebuild_features_and_forecast(raw_df_scenario, shock_idx):
        d = raw_df_scenario.copy()
        d["Kalman"] = kalman_filter_manual(d["Close"].values)
        d["Wavelet_causal"] = wavelet_haar_causal(d["Close"].values)
        for col in macro_cols:
            d[f"{col}_weekly"] = d[col].resample("W").last().reindex(d.index, method="ffill")
            d[f"{col}_monthly"] = d[col].resample("ME").last().reindex(d.index, method="ffill")
        d_lstm = d[FEATURES_DAILY].copy()
        d_lstm["log_return"] = np.log(d_lstm[TARGET_COL] / d_lstm[TARGET_COL].shift(1))
        for col in PRICE_LEVEL_COLS:
            d_lstm[col] = d_lstm[col].pct_change()
        for col in RATE_LEVEL_COLS:
            d_lstm[col] = d_lstm[col].diff()
        d_lstm.replace([np.inf, -np.inf], np.nan, inplace=True)
        d_lstm.dropna(inplace=True)

        X_sc = scaler_X.transform(d_lstm[feature_cols])
        X_seq_s, _ = build_sequences(X_sc, np.zeros((len(X_sc), 1)), LOOKBACK)
        start_win = max(shock_idx - LOOKBACK, 0)
        if start_win + HORIZON > len(X_seq_s):
            start_win = max(len(X_seq_s) - HORIZON, 0)

        last_price = d_lstm[TARGET_COL].iloc[start_win + LOOKBACK - 1]
        true_prices = d_lstm[TARGET_COL].iloc[start_win + LOOKBACK: start_win + LOOKBACK + HORIZON].values
        preds = recursive_forecast_corrected(model, X_seq_s[start_win], last_price)
        n = min(len(true_prices), len(preds))
        return true_prices[:n], preds[:n]

    raw_context = df.copy()
    fixed_anchors = [raw_context.index.get_indexer([d], method="nearest")[0] for d in FIXED_ANCHOR_DATES]

    def evaluate_scenario(shock_fn, label, anchors, **kwargs):
        mapes, rates = [], []
        for anchor in anchors:
            np.random.seed(int(anchor))
            shocked_raw, shock_idx = shock_fn(raw_context, int(anchor), **kwargs)
            true_p, preds_p = rebuild_features_and_forecast(shocked_raw, shock_idx)
            mapes.append(mean_absolute_percentage_error(true_p, preds_p) * 100)
            lo, hi = conformal_interval(preds_p, ALERT_CONFIDENCE)
            rates.append(risk_alert_flags(true_p, lo, hi).mean() * 100)
        print(f"{label}: MAPE mean={np.mean(mapes):.2f}% (min={np.min(mapes):.2f}, max={np.max(mapes):.2f}) "
              f"| alert rate mean={np.mean(rates):.1f}%")
        return mapes, rates

    print("\n--- Stress scenarios (5 fixed anchors) ---")
    mapes_policy, rates_policy = evaluate_scenario(apply_policy_shock, "Policy shock", fixed_anchors)
    mapes_crash, rates_crash = evaluate_scenario(apply_market_crash, "Market crash", fixed_anchors)

    # Illustrative single-anchor plots
    for label, shock_fn, mapes_out, fname in [
        ("Policy shock", apply_policy_shock, mapes_policy, "fig12_stress_scenario_policy_shock.png"),
        ("Market crash", apply_market_crash, mapes_crash, "fig12_stress_scenario_market_crash.png"),
    ]:
        anchor = fixed_anchors[2]  # 2024-04-11, Volatile regime
        np.random.seed(int(anchor))
        shocked_raw, shock_idx = shock_fn(raw_context, int(anchor))
        true_p, preds_p = rebuild_features_and_forecast(shocked_raw, shock_idx)
        lo80, hi80 = conformal_interval(preds_p, 0.80)
        fig, ax = plt.subplots(figsize=(12, 5))
        x_axis = np.arange(len(true_p))
        ax.fill_between(x_axis, lo80, hi80, alpha=0.3, color="orange", label="80% interval")
        ax.plot(x_axis, true_p, "k--o", label="Realized price (shocked)")
        ax.plot(x_axis, preds_p, color="steelblue", label="Bias-corrected LSTM forecast")
        ax.set_title(f"Stress scenario: {label}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES, fname), dpi=150, bbox_inches="tight")
        plt.close()

    # --- Robustness summary ---
    baseline_mape = mean_absolute_percentage_error(df_bt["true"], df_bt["pred"]) * 100
    robustness_data = {
        "Scenario": ["Normal market (all)"] + df_regime["Regime"].tolist() + ["Policy shock", "Market crash"],
        "MAPE (%)": [round(baseline_mape, 2)] + df_regime["LSTM_MAPE"].tolist() +
                    [round(np.mean(mapes_policy), 2), round(np.mean(mapes_crash), 2)],
    }
    df_robust = pd.DataFrame(robustness_data)
    print("\nRobustness summary:")
    print(df_robust.to_string(index=False))
    df_robust.to_csv(os.path.join(DATA, "phase4_robustness_table.csv"), index=False)

    angles = np.linspace(0, 2 * np.pi, len(df_robust), endpoint=False).tolist()
    values = df_robust["MAPE (%)"].tolist()
    values_plot, angles_plot = values + values[:1], angles + angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles_plot, values_plot, color="steelblue", linewidth=2)
    ax.fill(angles_plot, values_plot, color="steelblue", alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(df_robust["Scenario"].tolist(), fontsize=9)
    ax.axhline(8, color="red", linestyle="--")
    ax.set_title("Robustness radar (8% threshold in red)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig13_robustness_radar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Risk warning indicator, by regime ---
    alert_records = []
    for w in range(n_test_windows):
        start = w * HORIZON
        if start + HORIZON > len(X_test):
            break
        idx_full = split_idx + start
        last_price_w = data_lstm[TARGET_COL].iloc[idx_full + LOOKBACK - 1]
        true_w = data_lstm[TARGET_COL].iloc[idx_full + LOOKBACK: idx_full + LOOKBACK + HORIZON].values
        window_dates = data_lstm.index[idx_full + LOOKBACK: idx_full + LOOKBACK + HORIZON]
        preds_w = recursive_forecast_corrected(model, X_test[start], last_price_w)
        lo, hi = conformal_interval(preds_w, ALERT_CONFIDENCE)
        flags = risk_alert_flags(true_w, lo, hi)
        for d, f in zip(window_dates, flags):
            alert_records.append({"date": d, "regime": regime.get(d, "Normal"), "alert": f})
    df_alert = pd.DataFrame(alert_records)
    alert_by_regime = df_alert.groupby("regime")["alert"].mean() * 100

    print("\n" + "=" * 60)
    print("PHASE 4 -- SUMMARY")
    print("=" * 60)
    print("\nRegime MAPE:")
    print(df_regime.to_string(index=False))
    print("\nAlert rate by regime (%):")
    print(alert_by_regime.round(1))
    print(f"\nStress scenario alert rates: policy={np.mean(rates_policy):.1f}%, crash={np.mean(rates_crash):.1f}%")
    print(f"\nAll conditions stay under the project's 8% MAPE threshold: "
          f"{'YES' if df_robust['MAPE (%)'].max() < 8 else 'NO -- see table above'}")
    print(f"\nFigures saved to {FIGURES}")


if __name__ == "__main__":
    main()
