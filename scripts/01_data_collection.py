"""
BYD Stock Volatility Analysis -- Step 1: Data Collection

Builds the full dataset: BYD OHLCV + 9 external factors, aligned on BYD's
trading calendar (forward-fill only, no interpolation -- avoids leakage).

If data/byd_all_factors_dataset.csv already exists (committed to the repo),
it is loaded as-is and nothing is re-downloaded. Otherwise this re-fetches
everything live via akshare / yfinance / FRED / Open-Meteo, which will
produce a dataset with more recent rows than the one used in the report.

Run: python 01_data_collection.py
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
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

plt.rcParams.update({
    "figure.dpi": 120, "figure.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
})


# --------------------------------------------------------------------------
# Project paths (no Colab, no Drive)
# --------------------------------------------------------------------------
def find_project_root(markers=("data", "models")):
    here = os.getcwd()
    for path in (here, os.path.dirname(here)):
        if all(os.path.isdir(os.path.join(path, m)) for m in markers):
            return path
    return here


PROJECT = find_project_root()
DATA = os.path.join(PROJECT, "data")
FIGURES = os.path.join(PROJECT, "figures")
for folder in (DATA, FIGURES):
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

FINAL_CSV = os.path.join(DATA, "byd_all_factors_dataset.csv")
print(f"Project root: {PROJECT}")


# --------------------------------------------------------------------------
# Fetch functions (only called if FINAL_CSV is missing)
# --------------------------------------------------------------------------
def fetch_byd_price():
    import akshare as ak
    try:
        raw = ak.stock_zh_a_hist(
            symbol="002594", period="daily",
            start_date=START_DATE.replace("-", ""), end_date=END_DATE.replace("-", ""),
            adjust="hfq",
        )
        raw.rename(columns={"日期": "Date", "开盘": "Open", "收盘": "Close",
                             "最高": "High", "最低": "Low", "成交量": "Volume"}, inplace=True)
        raw["Date"] = pd.to_datetime(raw["Date"])
        raw.set_index("Date", inplace=True)
        raw.sort_index(inplace=True)
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        print(f"BYD price via akshare: {len(df)} rows")
    except Exception as e:
        print(f"akshare failed ({e}), falling back to yfinance")
        import yfinance as yf
        df = yf.download("002594.SZ", start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)[["Open", "High", "Low", "Close", "Volume"]]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    df = df.ffill(limit=2).dropna()
    df["DailyReturn"] = df["Close"].pct_change()
    return df


def fetch_external_factors(byd_index):
    import yfinance as yf
    import pandas_datareader.data as web
    import akshare as ak
    import requests

    factors = {}

    tickers = {"SP500": "^GSPC", "VIX": "^VIX", "Tesla": "TSLA",
               "Brent_Oil": "BZ=F", "Copper": "HG=F", "SSE": "000001.SS"}
    raw = yf.download(list(tickers.values()), start=START_DATE, end=END_DATE,
                       auto_adjust=True, progress=False)
    for name, ticker in tickers.items():
        try:
            factors[name] = raw["Close"][ticker]
        except Exception as e:
            print(f"  {name}: {e}")

    try:
        cnyusd = web.DataReader("DEXCHUS", "fred", start=START_DATE, end=END_DATE)
        factors["CNYUSD"] = cnyusd.iloc[:, 0]
    except Exception as e:
        print(f"CNY/USD (FRED) failed: {e}")

    try:
        raw = ak.stock_zh_a_hist(symbol="300750", period="daily",
                                  start_date=START_DATE.replace("-", ""), end_date=END_DATE.replace("-", ""),
                                  adjust="hfq")
        raw.rename(columns={"日期": "Date", "收盘": "CATL_Close"}, inplace=True)
        raw["Date"] = pd.to_datetime(raw["Date"])
        raw.set_index("Date", inplace=True)
        factors["CATL_Close"] = raw["CATL_Close"]
    except Exception as e:
        print(f"CATL failed: {e}")

    try:
        raw = ak.macro_china_lpr()
        date_col = raw.columns[0]
        rate_col = [c for c in raw.columns if "1" in str(c)][0]
        raw[date_col] = pd.to_datetime(raw[date_col])
        raw.set_index(date_col, inplace=True)
        factors["PBoC_LPR_1y"] = pd.to_numeric(raw[rate_col], errors="coerce")
    except Exception as e:
        print(f"PBoC LPR failed: {e}")

    try:
        url = ("https://archive-api.open-meteo.com/v1/archive"
               "?latitude=39.9042&longitude=116.4074"
               f"&start_date={START_DATE}&end_date=2024-12-31"
               "&daily=temperature_2m_mean&timezone=Asia%2FShanghai")
        data = requests.get(url, timeout=30).json()
        factors["Beijing_Temp_C"] = pd.Series(
            data["daily"]["temperature_2m_mean"],
            index=pd.to_datetime(data["daily"]["time"]))
    except Exception as e:
        print(f"Beijing temperature failed: {e}")

    df_factors = pd.DataFrame(factors)
    df_factors.index.name = "Date"
    df_factors = df_factors.reindex(byd_index).ffill()
    return df_factors


def build_dataset_from_scratch():
    print("No local dataset found -- fetching live data.")
    print("WARNING: results will differ from the report (more recent data, live APIs).")
    df_price = fetch_byd_price()
    df_factors = fetch_external_factors(df_price.index)
    df = df_price.join(df_factors, how="left").ffill()
    df.dropna(inplace=True)
    return df


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    if os.path.exists(FINAL_CSV):
        df = pd.read_csv(FINAL_CSV, parse_dates=["Date"], index_col="Date")
        df.sort_index(inplace=True)
        print(f"Loaded existing dataset: {df.shape} "
              f"({df.index.min().date()} to {df.index.max().date()})")
    else:
        df = build_dataset_from_scratch()
        df.to_csv(FINAL_CSV)
        print(f"Saved new dataset: {df.shape} -> {FINAL_CSV}")

    # --- Exploratory analysis ---
    print("\nMissing values:")
    for col in df.columns:
        pct = df[col].isnull().mean() * 100
        flag = "WARN" if pct > 5 else "ok"
        print(f"  [{flag}] {col:<20} {pct:.1f}%")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(df.index, df["Close"], color="#1a6fc4", linewidth=1.0)
    ax1.fill_between(df.index, df["Close"], alpha=0.08, color="#1a6fc4")
    ax1.set_ylabel("Close price (CNY)")
    ax1.set_title("BYD (002594.SZ) daily closing price, 2015-2025", fontweight="bold")
    ax2.bar(df.index, df["Volume"] / 1e6, color="#6baed6", alpha=0.6, width=1)
    ax2.set_ylabel("Volume (M shares)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig1_byd_price_volume.png"), dpi=150, bbox_inches="tight")
    plt.close()

    df["Vol_20d"] = df["DailyReturn"].rolling(20).std() * np.sqrt(252)
    df["Vol_60d"] = df["DailyReturn"].rolling(60).std() * np.sqrt(252)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax1.plot(df.index, df["Close"], color="#1a6fc4", linewidth=0.9)
    ax1.set_ylabel("Price (CNY)")
    ax1.set_title("BYD: price and realized volatility", fontweight="bold")
    ax2.plot(df.index, df["Vol_20d"], color="#e34234", linewidth=1.0, label="20d")
    ax2.plot(df.index, df["Vol_60d"], color="#f4a460", linewidth=1.0, label="60d")
    ax2.legend()
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig1_volatility.png"), dpi=150, bbox_inches="tight")
    plt.close()

    returns = df["DailyReturn"].dropna()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(returns, bins=80, color="#1a6fc4", alpha=0.7, density=True, edgecolor="white", linewidth=0.3)
    x = np.linspace(returns.quantile(0.001), returns.quantile(0.999), 200)
    ax1.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()), color="red", linestyle="--", label="Normal fit")
    ax1.set_title("Return distribution")
    ax1.legend()
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title("Q-Q plot")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig2_return_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSkewness: {returns.skew():.3f} | Kurtosis (excess): {returns.kurtosis():.3f}")

    # ADF stationarity
    adf_level = adfuller(df["Close"].dropna())
    adf_diff = adfuller(df["Close"].diff().dropna())
    print(f"\nADF (levels):    stat={adf_level[0]:.3f}, p={adf_level[1]:.4f}")
    print(f"ADF (1st diff):  stat={adf_diff[0]:.3f}, p={adf_diff[1]:.4f}")

    # FFT cyclicality
    r = returns.values
    fft_power = np.abs(np.fft.rfft(r - r.mean())) ** 2
    fft_freq = np.fft.rfftfreq(len(r), d=1.0)
    periods = np.where(fft_freq > 0, 1 / fft_freq, np.inf)
    top_idx = np.argsort(fft_power[1:])[::-1][:5] + 1
    print("\nTop 5 FFT spectral peaks:")
    for i in top_idx:
        print(f"  period ~ {periods[i]:.1f} days, power = {fft_power[i]:.4f}")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fft_freq[1:], fft_power[1:], color="#1a6fc4", linewidth=0.8)
    ax.set_xlabel("Frequency (cycles / trading day)")
    ax.set_title("FFT power spectrum of BYD daily returns")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fft_return_spectrum.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Correlation heatmap (returns, not levels)
    numeric_cols = [c for c in df.columns if df[c].dtype in ("float64", "int64") and df[c].isnull().mean() < 0.2]
    corr = df[numeric_cols].pct_change().dropna().corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn", vmin=-1, vmax=1,
                square=True, linewidths=0.4, annot_kws={"size": 7})
    ax.set_title("Correlation matrix -- daily returns", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig3_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    if "Close" in corr.columns:
        print("\nTop factors correlated with BYD Close return:")
        print(corr["Close"].drop("Close").abs().sort_values(ascending=False).head(10).to_string())

    print(f"\nDone. Dataset ready at {FINAL_CSV} ({df.shape[0]} rows, {df.shape[1]} columns).")
    print(f"Figures saved to {FIGURES}")


    # Fig.4 -- BYD price vs external factors
    factors_available = []
    if 'CNYUSD' in df.columns:
        factors_available.append(('CNYUSD', 'CNY/USD exchange rate', '#2ca02c'))
    if 'SSE_Close' in df.columns:
        factors_available.append(('SSE_Close', 'Shanghai Composite', '#ff7f0e'))

    if factors_available:
        n = len(factors_available) + 1  # +1 for BYD price
        fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)

        axes[0].plot(df.index, df['Close'], color='#1a6fc4', linewidth=0.8)
        axes[0].set_ylabel('BYD Close (CNY)')
        axes[0].set_title('BYD stock price vs external factors', fontweight='bold')

        for i, (col, label, color) in enumerate(factors_available, start=1):
            axes[i].plot(df.index, df[col], color=color, linewidth=0.8)
            axes[i].set_ylabel(label, fontsize=9)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES, "fig4_factors_overview.png"), dpi=150, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
