# --- Install required libraries ---
!pip install yfinance pandas numpy matplotlib scipy --quiet

# --- Imports ---
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.fft import fft
import matplotlib.pyplot as plt

# --- Dominant τ(x) computation using FFT ---
def dominant_tau(series):
    series = np.asarray(series.values, dtype=np.float64)
    freqs = np.fft.fftfreq(len(series))
    amps = np.abs(fft(series - np.mean(series)))
    mask = (freqs > 0.1) & (freqs < 1.0)
    if np.any(mask):
        peak = freqs[mask][np.argmax(amps[mask])]
        return 1 / peak if peak != 0 else np.inf
    return np.inf

# === FINANCE DATA EXAMPLE ===
# Download daily closing prices for AAPL, MSFT, GOOGL
tickers = ["AAPL", "MSFT", "GOOGL"]
df_finance = yf.download(tickers, start="2024-04-01", end="2024-05-16")["Close"]
df_finance = df_finance.fillna(method='ffill')

# Binarize trends: 1 if daily price increased, else 0
df_fin_bin = (df_finance.diff() > 0).astype(int)

# Compute τ(x) for each ticker
taus_fin = {col: dominant_tau(df_fin_bin[col]) for col in df_fin_bin.columns}

# Variance between adjacent τ-values
var_msft = np.var([taus_fin["AAPL"], taus_fin["MSFT"]])
var_googl = np.var([taus_fin["MSFT"], taus_fin["GOOGL"]])

# Causal probability (Theorem formulation)
p_fin = np.exp(-var_googl) / (np.exp(-var_googl) + np.exp(-0.5))

# === COVID DATA EXAMPLE ===
# Download COVID-19 smoothed daily case counts
url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
df_covid = pd.read_csv(url, usecols=["location", "date", "new_cases_smoothed"])
df_covid["date"] = pd.to_datetime(df_covid["date"])
df_covid = df_covid[df_covid["location"].isin(["Italy", "Germany", "France"])]
df_covid = df_covid[df_covid["date"].between("2022-10-01", "2022-11-15")]
df_covid = df_covid.pivot(index="date", columns="location", values="new_cases_smoothed").fillna(0)

# Binarize trends: 1 if daily cases increased, else 0
df_covid_bin = (df_covid.diff() > 0).astype(int)

# Compute τ(x) for each country
taus_covid = {col: dominant_tau(df_covid_bin[col]) for col in df_covid_bin.columns}

# Variance between adjacent τ-values
var_it_ger = np.var([taus_covid["Italy"], taus_covid["Germany"]])
var_ger_fr = np.var([taus_covid["Germany"], taus_covid["France"]])

# Causal probability (Theorem formulation)
p_covid = np.exp(-var_ger_fr) / (np.exp(-var_ger_fr) + np.exp(-0.5))

# === OUTPUT ===
print("=== Financial Data Example ===")
print(f"τ-values: {taus_fin}")
print(f"Var(τ[MSFT, GOOGL]): {var_googl:.4f}")
print(f"P(GOOGL = 1 | AAPL = 1) ≈ {p_fin:.3f}")

print("\n=== COVID-19 Data Example ===")
print(f"τ-values: {taus_covid}")
print(f"Var(τ[Germany, France]): {var_ger_fr:.4f}")
print(f"P(France = 1 | Italy = 1) ≈ {p_covid:.3f}")
