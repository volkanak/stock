import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
from PIL import Image
import pandas_ta as ta
import datetime
from datetime import datetime, date, time, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

from prophet import Prophet



import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')
plt.style.use('fivethirtyeight')



def SMA(data, period=30, column="Close"):
    return data[column].rolling(window=period).mean()

def EMA(data, period=21, column="Close"):
    return data[column].ewm(span=period, adjust=False).mean()

def MACD(data, period_long=26, period_short=12, period_signal=9, column="Close"):
    ShortEMA=EMA(data, period_short, column=column)
    LongEMA=EMA(data, period_long, column=column)
    data["MACD"] = ShortEMA - LongEMA
    data["Signal_Line"] = EMA(data, period_signal, column="MACD")
    st.write(data)
    return data

def RSI(data, period=14, column="Close"):
    delta = data[column].diff(1)
    delta = delta[1:]
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    data["up"] = up
    data["down"] = down
    AVG_Gain = SMA(data, period, column="up")
    AVG_Loss = abs(SMA(data, period, column="down"))
    RS = AVG_Gain / AVG_Loss
    RSI = 100.0 - (100.0 / (1.0 + RS))
    data["RSI"] = RSI
    return data

def find_head_and_shoulders(data, column="Close"):
    peaks = data[column][(data[column].shift(1) < data[column]) & (data[column].shift(-1) < data[column])]
    troughs = data[column][(data[column].shift(1) > data[column]) & (data[column].shift(-1) > data[column])]

    hns = []
    for i in range(1, len(peaks) - 1):
        if peaks.index[i] < troughs.index[i] < peaks.index[i+1] and peaks.index[i+1] < troughs.index[i+1] < peaks.index[i+2]:
            if peaks[i] < peaks[i+1] > peaks[i+2]:
                hns.append((peaks.index[i], peaks.index[i+1], peaks.index[i+2]))

    return hns

# Define Ichimoku calculation and plot function
def ichimoku_indicators(data):
    # Calculate Ichimoku components
    tenkan_sen = (data['High'].rolling(window=9).max() + data['Low'].rolling(window=9).min()) / 2
    kijun_sen = (data['High'].rolling(window=26).max() + data['Low'].rolling(window=26).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((data['High'].rolling(window=52).max() + data['Low'].rolling(window=52).min()) / 2).shift(26)
    chikou_span = data['Close'].shift(-26)

    # Plot Ichimoku
    plt.figure(figsize=(10, 6))
    plt.title(f"Ichimoku Indicators for {secim}")
    plt.plot(data['Close'], label='Close', color='grey', alpha=0.6)
    plt.plot(data.index, tenkan_sen, label='Tenkan Sen', color='red')
    plt.plot(data.index, kijun_sen, label='Kijun Sen', color='blue')
    plt.fill_between(data.index, senkou_span_a, senkou_span_b, where=senkou_span_a >= senkou_span_b, color='lightgreen', alpha=0.3)
    plt.fill_between(data.index, senkou_span_a, senkou_span_b, where=senkou_span_a < senkou_span_b, color='lightcoral', alpha=0.3)
    plt.legend()
    st.pyplot(plt)



def plot_supertrend_and_fibonacci(stock_symbol):
    
    data = yf.download(stock_symbol, start='2022-01-01', end='2024-04-30')

  
    supertrend = ta.supertrend(data['High'], data['Low'], data['Close'], length=7, multiplier=3.0)
    data = data.join(supertrend)

   
    data['Buy_Signal'] = (data['SUPERTd_7_3.0'] == 1)
    data['Sell_Signal'] = (data['SUPERTd_7_3.0'] == -1)

    # Plotting the data
    plt.figure(figsize=(14,7))
    plt.plot(data['Close'], label='Kapanış Fiyatı', alpha=0.5)
    plt.plot(data['SUPERT_7_3.0'], label='SuperTrend', alpha=0.5)
    plt.scatter(data.index[data['Buy_Signal']], data['Close'][data['Buy_Signal']], label='Al', marker='^', color='green', s=100)
    plt.scatter(data.index[data['Sell_Signal']], data['Close'][data['Sell_Signal']], label='Sat', marker='v', color='red', s=100)

    # Fibonacci Retracement Levels
    max_price = data['Close'].max()
    min_price = data['Close'].min()
    diff = max_price - min_price
    fibo_levels = [0, 0.236, 0.382, 0.5, 0.618, 1]
    for level in fibo_levels:
        price = min_price + diff * level
        plt.axhline(y=price, color='blue', linestyle='--', alpha=0.5)
        plt.text(data.index[-1], price, f"{price:.2f}", fontsize=12, color='blue')

    plt.title('SuperTrend ve Fibonacci Düzeltme Seviyeleri')
    plt.legend()
    st.pyplot(plt)








secim='AAPL'

st.title('Stock Market Analysis')

st.sidebar.image('logo/borsa.jpeg',width=70)
st.sidebar.title('Stock Market Filter')

islemturu=st.sidebar.radio('İşlem Türü',('Kripto','Borsa'))
st.write('Seçilen İşlem Türü:',islemturu)
if islemturu=='Kripto':
    kriptosec=st.sidebar.selectbox('Kripto Seçiniz',('BTC','ETH','ADA'))
    kriptosec=kriptosec+'-USD'
    secim=kriptosec

else:
    borsasec=st.sidebar.selectbox('Borsa Seçiniz',('Apple','Google','Microsoft'))
    senetler ={
      'Apple':'AAPL',
      'Google':'GOOGL',
      'Microsoft':'MSFT',
      'Tesla':'TSLA'
    }
    borsasec=senetler[borsasec]
    secim=borsasec

zaralik=range(1,4500)
bugun=datetime.today()
prophet=st.sidebar.checkbox('Borsa Tahmini')
# if prophet:
#     paralik=st.sidebar.number_input('Borsa tahmin aralik',min_value=0.0,max_value=100,value=30)

slider=st.sidebar.select_slider('Zaman Aralığı',options=zaralik,value=30)
aralik=timedelta(days=slider)
start=st.sidebar.date_input('Başlangıç Tarihi',value=bugun-aralik)   

end=st.sidebar.date_input('Bitiş Tarihi',value=bugun)
data=yf.download(secim,start=start,end=end)




def get_data(secim):
    data=yf.download(secim,start=start,end=end)
    st.line_chart(data['Close'])
    st.line_chart(data['Volume'])

    if prophet:
        data['ds']=data.index
        data['y']=data['Close']
        model=Prophet()
        model.fit(data)
        future=model.make_future_dataframe(periods=30)
        forecast=model.predict(future)
        st.write(forecast)
        fig1=model.plot(forecast)
        st.pyplot(fig1)
        fig2=model.plot_components(forecast)
        st.pyplot(fig2)
    
get_data(secim)


st.sidebar.write("### Finansal İndikatörler")
fi = st.sidebar.checkbox("Finansal İndikatörler")



if fi:
    fimacd = st.sidebar.checkbox("MACD")
    fimacd2 = st.sidebar.checkbox("MACD&Signal Line")
    firsi = st.sidebar.checkbox("RSI")
    fisl = st.sidebar.checkbox("Signal Line")
    fho = st.sidebar.checkbox("Omuz-Baş-Omuz")
    show_ichimoku = st.sidebar.checkbox("Ichimoku Indicators")
    supertrend = st.sidebar.checkbox("Super Trend and Fibonacci Indicators")

    if fimacd:
        macd = MACD(data)
        plt.plot(macd.index, macd["MACD"], label="MACD", color='green')
        st.write("MACD")
        st.line_chart(macd["MACD"])
    
    if supertrend:
        plot_supertrend_and_fibonacci(secim)
    

    if show_ichimoku:
        ichimoku_indicators(data)

    if fimacd2:
        macd = MACD(data)
        plt.figure(figsize=(12, 6))
        plt.plot(macd.index, macd["MACD"], label="MACD", color='blue')
        plt.plot(macd.index, macd["Signal_Line"], label="Signal Line", color='red')
        plt.legend(loc="upper left")
        st.pyplot(plt)
    
    if firsi:
        rsi = RSI(data)
        plt.figure(figsize=(12, 6))
        plt.plot(rsi.index, rsi["RSI"], label="RSI", color='green')
        plt.axhline(y=70, color='r', linestyle='--', label='Overbought')
        plt.axhline(y=30, color='b', linestyle='--', label='Oversold')
        plt.legend(loc="upper left")
        st.pyplot(plt)
        st.table(rsi)

    if fisl:
        macd = MACD(data)
        st.write("Signal Line")
        st.line_chart(macd["Signal_Line"])
    if fho:
        hns = find_head_and_shoulders(data)
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data["Close"], label="Close", color='black')
        for h in hns:
            plt.plot([h[0], h[1], h[2]], [data["Close"][h[0]], data["Close"][h[1]], data["Close"][h[2]]], color='orange')
            plt.scatter([h[0], h[1], h[2]], [data["Close"][h[0]], data["Close"][h[1]], data["Close"][h[2]]], color='orange')
        plt.legend(loc="upper left")
        st.pyplot(plt)
        st.table(hns)
       



