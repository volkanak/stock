import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import yfinance as yf
from scipy.integrate import simps

# SMA ve EMA fonksiyonları
def SMA(data, period=30, column="Close"):
    return data[column].rolling(window=period).mean()

def EMA(data, period=21, column="Close"):
    return data[column].ewm(span=period, adjust=False).mean()

def MACD(data, period_long=26, period_short=12, period_signal=9, column="Close"):
    ShortEMA = EMA(data, period_short, column=column)
    LongEMA = EMA(data, period_long, column=column)
    data["MACD"] = ShortEMA - LongEMA
    data["Signal_Line"] = EMA(data, period_signal, column="MACD")
    return data

# Verileri indirin
data = yf.download('AAPL', start='2020-01-01', end='2024-05-01')







from scipy.signal import find_peaks
from scipy.integrate import simps




# EMA fonksiyonları
def EMA(data, period=21, column="Close"):
    return data[column].ewm(span=period, adjust=False).mean()

def MACD(data, period_long=26, period_short=12, period_signal=9, column="Close"):
    ShortEMA = EMA(data, period_short, column=column)
    LongEMA = EMA(data, period_long, column=column)
    data["MACD"] = ShortEMA - LongEMA
    data["Signal_Line"] = EMA(data, period_signal, column="MACD")
    return data

# MACD hesaplama
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

# Ortalama zirve aralığını gün sayısı olarak bulun
average_interval_days = np.mean([interval.days for interval in peak_intervals])

# Son zirve zamanını ve bir sonraki tahmini zirve zamanını bulun
last_peak_time = peak_times[-1]
next_peak_time = last_peak_time + pd.Timedelta(days=average_interval_days)

# Tahmin edilen bir sonraki zirve değerini bulmak için MACD verilerini kullanarak alanı hesapla
next_peak_index = macd.index.get_loc(next_peak_time, method='nearest')
macd_segment = macd_values[peaks[-1]:next_peak_index]
time_segment = macd.index[peaks[-1]:next_peak_index]

# Alanı hesapla
area = simps(macd_segment, dx=1)

# Grafikle gösterin
plt.figure(figsize=(12, 6))
plt.plot(macd.index, macd["MACD"], label='MACD', color='blue')
plt.plot(macd.index, macd["Signal_Line"], label='Signal Line', color='red')
plt.plot(peak_times, peak_values, 'x', label='Peaks', color='green')
plt.axvline(next_peak_time, color='orange', linestyle='--', label='Next Peak (Predicted)')
plt.legend(loc='upper left')
plt.show()

average_interval_days, area




# MACD verilerini alın
macd_values = macd["MACD"].values

# Zirveleri tespit edin
peaks, _ = find_peaks(macd_values, distance=30)

# Zirve zamanlarını ve değerlerini alın
peak_times = macd.index[peaks]
peak_values = macd_values[peaks]

# Zirve aralıklarını hesaplayın
peak_intervals = np.diff(peak_times)

# Ortalama zirve aralığını gün sayısı olarak bulun
average_interval_days = np.mean([interval.days for interval in peak_intervals])

print(f"Ortalama zirve aralığı (gün): {average_interval_days}")

# Son zirve zamanını ve bir sonraki tahmini zirve zamanını bulun
last_peak_time = peak_times[-1]
next_peak_time = last_peak_time + pd.Timedelta(days=average_interval_days)

# Tahmin edilen bir sonraki zirve değerini bulmak için MACD verilerini kullanarak alanı hesapla
next_peak_index = macd.index.get_loc(next_peak_time, method='nearest')
macd_segment = macd_values[peaks[-1]:next_peak_index]
time_segment = macd.index[peaks[-1]:next_peak_index]

# Alanı hesapla
area = simps(macd_segment, dx=1)
print(f"Tahmin edilen alan: {area}")

# Grafikle gösterin
plt.figure(figsize=(12, 6))
plt.plot(macd.index, macd["MACD"], label='MACD', color='blue')
plt.plot(macd.index, macd["Signal_Line"], label='Signal Line', color='red')
plt.plot(peak_times, peak_values, 'x', label='Peaks', color='green')
plt.axvline(next_peak_time, color='orange', linestyle='--', label='Next Peak (Predicted)')
plt.legend(loc='upper left')
plt.show()
