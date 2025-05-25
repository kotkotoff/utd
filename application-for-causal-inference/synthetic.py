import numpy as np
import pandas as pd
from scipy.fft import fft

np.random.seed(42)
n = 46  # с 2025-04-01 по 2025-05-16
t = np.arange(n)
cities = {
    "London": 15 + 5 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 1, n),
    "Paris": 16 + 5 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 1, n),
    "Berlin": 14 + 5 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 1, n)
}
data = pd.DataFrame(cities)

def dominant_tau(signal):
    signal = np.asarray(signal, dtype=np.float64)
    spectrum = np.abs(fft(signal - np.mean(signal)))
    freqs = np.fft.fftfreq(len(signal))
    mask = (freqs > 0.1) & (freqs < 1.0)
    peak_freq = freqs[mask][np.argmax(spectrum[mask])]
    return 1 / peak_freq if peak_freq != 0 else np.inf

taus = {city: dominant_tau(data[city]) for city in data.columns}


var_lp = np.var([taus["London"], taus["Paris"]])
var_pb = np.var([taus["Paris"], taus["Berlin"]])

numerator = np.exp(-var_pb)
denominator = numerator + np.exp(-0.5)
prob = numerator / denominator

print(f"Tau values: {taus}")
print(f"Var(London, Paris): {var_lp:.4f}")
print(f"Var(Paris, Berlin): {var_pb:.4f}")
print(f"P(Berlin = 1 | London = 1) = {prob:.3f}")
