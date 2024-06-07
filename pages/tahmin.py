import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import yfinance as yf





def SMA(data, period=30, column="Close"):
    return data[column].rolling(window=period).mean()

def EMA(data, period=21, column="Close"):
    return data[column].ewm(span=period, adjust=False).mean()

def MACD(data, period_long=26, period_short=12, period_signal=9, column="Close"):
    ShortEMA=EMA(data, period_short, column=column)
    LongEMA=EMA(data, period_long, column=column)
    data["MACD"] = ShortEMA - LongEMA
    data["Signal_Line"] = EMA(data, period_signal, column="MACD")
    return data

# Verileri indirin
data = yf.download('AAPL', start='2020-01-01', end='2024-05-01')
macd = MACD(data)

# MACD verilerini alın
macd_values = macd["MACD"].values

# Zirveleri tespit edin
peaks, _ = find_peaks(macd_values, distance=30)

# Zirve zamanlarını ve değerlerini alın
peak_times = macd.index[peaks]
peak_values = macd_values[peaks]

# Zirve aralıklarını hesaplayın
peak_intervals = np.diff(peak_times)

# Ortalama zirve aralığını bulun
average_interval = peak_intervals.mean()

print(f"Ortalama zirve aralığı: {average_interval}")

# Grafikle gösterin
plt.figure(figsize=(12, 6))
plt.plot(macd.index, macd["MACD"], label='MACD', color='blue')
plt.plot(macd.index, macd["Signal_Line"], label='Signal Line', color='red')
plt.plot(peak_times, peak_values, 'x', label='Peaks', color='green')
plt.legend(loc='upper left')
plt.show()
