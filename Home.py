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
import os
import time
from bs4 import BeautifulSoup
import re
import requests
import base64
import json
import langchain
from langchain.agents import Tool, initialize_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.callbacks import StreamlitCallbackHandler
import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')
plt.style.use('fivethirtyeight')

# Existing functions for stock analysis
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

def ichimoku_indicators(data):
    tenkan_sen = (data['High'].rolling(window=9).max() + data['Low'].rolling(window=9).min()) / 2
    kijun_sen = (data['High'].rolling(window=26).max() + data['Low'].rolling(window=26).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((data['High'].rolling(window=52).max() + data['Low'].rolling(window=52).min()) / 2).shift(26)
    chikou_span = data['Close'].shift(-26)

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
    data = yf.download(stock_symbol, start='2022-01-01', end='2024-06-30')
    supertrend = ta.supertrend(data['High'], data['Low'], data['Close'], length=7, multiplier=3.0)
    data = data.join(supertrend)
    data['Buy_Signal'] = (data['SUPERTd_7_3.0'] == 1)
    data['Sell_Signal'] = (data['SUPERTd_7_3.0'] == -1)

    plt.figure(figsize=(14,7))
    plt.plot(data['Close'], label='Kapanış Fiyatı', alpha=0.5)
    plt.plot(data['SUPERT_7_3.0'], label='SuperTrend', alpha=0.5)
    plt.scatter(data.index[data['Buy_Signal']], data['Close'][data['Buy_Signal']], label='Al', marker='^', color='green', s=100)
    plt.scatter(data.index[data['Sell_Signal']], data['Close'][data['Sell_Signal']], label='Sat', marker='v', color='red', s=100)

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

secim = 'AAPL'



# st.sidebar.image('logo/borsa.jpeg', width=70)
st.sidebar.title('Stock Market Filter')






kriptosec = st.sidebar.selectbox('Hisse Seçiniz', (
        'Apple', 'Microsoft', 'Amazon', 'Alphabet (Google)', 'Meta (Facebook)', 'Tesla', 
        'NVIDIA', 'PayPal', 'Intel', 'Comcast', 'PepsiCo', 'Adobe', 'Cisco', 
        'Netflix', 'Amgen', 'Texas Instruments', 'Broadcom', 'Costco', 'Qualcomm', 
        'T-Mobile US', 'Charter Communications', 'Starbucks', 'Intuit', 
        'Advanced Micro Devices (AMD)', 'Mondelez', 'Intuitive Surgical', 'Booking Holdings', 
        'Lam Research', 'Gilead Sciences', 'Fiserv', 'Automatic Data Processing (ADP)', 'CSX', 
        'MercadoLibre', 'JD.com', 'Micron Technology', 'Analog Devices', 'Applied Materials', 
        'Illumina', 'Lululemon', 'Zoom Video Communications', 'DocuSign', 'ASML Holding', 
        'Regeneron Pharmaceuticals', 'Vertex Pharmaceuticals', 'KLA Corporation', 'Synopsys', 
        'Moderna', 'DexCom', 'Workday', 'Baidu', 'Monster Beverage', 'Align Technology', 
        'Cadence Design Systems', 'Atlassian', 'OReilly Automotive', 'PACCAR', 'Xcel Energy', 
        'Kraft Heinz', 'Skyworks Solutions', 'Altimeter Growth'
    ))
   

# else:
#     borsasec = st.sidebar.selectbox('Hisse Seçiniz', (
#         'Marathon Digital Holdings', 'Anadolu Efes', 'Agroland', 'Aygaz', 'Akbank', 'Akçansa', 
#         'Akfen GYO', 'Akfen Yenilenebilir Enerji', 'Aksa', 'Akiş GYO', 'Alarko Holding', 
#         'Albaraka Türk', 'Alfa Solar Enerji', 'Anadolu Sigorta', 'Arçelik', 'Aselsan', 
#         'Astor Enerji', 'Bera Holding', 'Bosch Fren', 'Bien Yapı', 'Bim Birleşik Mağazalar', 
#         'Biotrend Çevre ve Enerji', 'Borsa İstanbul', 'Borusan Yatırım', 'Brisa Bridgestone Sabancı', 
#         'Batıçim', 'Çanakkale Çimento', 'Coca Cola İçecek', 'Çimsa Çimento', 'CW Enerji', 
#         'Doğuş Otomotiv', 'Doğan Holding', 'Eczacıbaşı İlaç', 'Eczacıbaşı Yatırım', 'Ege Endüstri', 
#         'Emlak Konut', 'Enerjisa Enerji', 'Enka İnşaat', 'Ereğli Demir ve Çelik', 'EuroPower Enerji', 
#         'Euro Yatırım Holding', 'Ford Otosan', 'Garanti Bankası', 'Girişim Elektrik', 'Gübre Fabrikaları', 
#         'Galata Wind', 'Halkbank', 'Hektaş', 'İpek Doğal Enerji', 'İş Bankası', 'İş GYO', 
#         'İş Yatırım', 'İzmir Demir Çelik', 'Kayseri Şeker', 'Kardemir Karabük', 'Koç Holding', 
#         'Klimasan', 'Kontrolmatik Teknoloji', 'Konya Çimento', 'Koza Anadolu', 'Koza Altın', 
#         'Kardemir', 'Mavi Giyim', 'Migros', 'Mia Teknoloji', 'Odaş Elektrik', 'Otokar', 
#         'Oyak Çimento', 'Petkim', 'Pegasus', 'Quagr', 'Reeder Teknoloji', 'Sabancı Holding', 
#         'Sasa Polyester', 'Say Yenilenebilir Enerji', 'SDT Uzay ve Savunma', 'Şişecam', 
#         'Şekerbank', 'Smart Güneş Enerjisi', 'Şok Marketler', 'Tab Gıda', 'TAV Havalimanları', 
#         'Turkcell', 'Türk Hava Yolları', 'Tekfen', 'Tofaş', 'TSKB', 'Türk Telekom', 
#         'Türk Traktör', 'Tukaş', 'Tüpraş', 'Türk Sigorta', 'Ülker', 'VakıfBank', 
#         'Vestel Beyaz Eşya', 'Vestel', 'Yeo Teknoloji', 'Yapı Kredi', 'Yayla Agro', 'Zorlu Enerji'
#     ))
#     senetler = {
#         'Apple': 'AAPL',
#         'Microsoft': 'MSFT',
#         'Amazon': 'AMZN',
#         'Alphabet (Google)': 'GOOGL',
#         'Meta (Facebook)': 'META',
#         'Tesla': 'TSLA',
#         'NVIDIA': 'NVDA',
#         'PayPal': 'PYPL',
#         'Intel': 'INTC',
#         'Comcast': 'CMCSA',
#         'PepsiCo': 'PEP',
#         'Adobe': 'ADBE',
#         'Cisco': 'CSCO',
#         'Netflix': 'NFLX',
#         'Amgen': 'AMGN',
#         'Texas Instruments': 'TXN',
#         'Broadcom': 'AVGO',
#         'Costco': 'COST',
#         'Qualcomm': 'QCOM',
#         'T-Mobile US': 'TMUS',
#         'Charter Communications': 'CHTR',
#         'Starbucks': 'SBUX',
#         'Intuit': 'INTU',
#         'Advanced Micro Devices (AMD)': 'AMD',
#         'Mondelez': 'MDLZ',
#         'Intuitive Surgical': 'ISRG',
#         'Booking Holdings': 'BKNG',
#         'Lam Research': 'LRCX',
#         'Gilead Sciences': 'GILD',
#         'Fiserv': 'FI',
#         'Automatic Data Processing (ADP)': 'ADP',
#         'CSX': 'CSX',
#         'MercadoLibre': 'MELI',
#         'JD.com': 'JD',
#         'Micron Technology': 'MU',
#         'Analog Devices': 'ADI',
#         'Applied Materials': 'AMAT',
#         'Illumina': 'ILMN',
#         'Lululemon': 'LULU',
#         'Zoom Video Communications': 'ZM',
#         'DocuSign': 'DOCU',
#         'ASML Holding': 'ASML',
#         'Regeneron Pharmaceuticals': 'REGN',
#         'Vertex Pharmaceuticals': 'VRTX',
#         'KLA Corporation': 'KLAC',
#         'Synopsys': 'SNPS',
#         'Moderna': 'MRNA',
#         'DexCom': 'DXCM',
#         'Workday': 'WDAY',
#         'Baidu': 'BIDU',
#         'Monster Beverage': 'MNST',
#         'Align Technology': 'ALGN',
#         'Cadence Design Systems': 'CDNS',
#         'Atlassian': 'TEAM',
#         'OReilly Automotive': 'ORLY',
#         'PACCAR': 'PCAR',
#         'Xcel Energy': 'XEL',
#         'Kraft Heinz': 'KHC',
#         'Skyworks Solutions': 'SWKS',
#         'Altimeter Growth': 'ALT',
#         'Marathon Digital Holdings': 'MARA',
#         'Anadolu Efes': 'AEFES.IS',
#         'Agroland': 'AGROT.IS',
#         'Aygaz': 'AHGAZ.IS',
#         'Akbank': 'AKBNK.IS',
#         'Akçansa': 'AKCNS.IS',
#         'Akfen GYO': 'AKFGY.IS',
#         'Akfen Yenilenebilir Enerji': 'AKFYE.IS',
#         'Aksa': 'AKSA.IS',
#         'Akiş GYO': 'AKSEN.IS',
#         'Alarko Holding': 'ALARK.IS',
#         'Albaraka Türk': 'ALBRK.IS',
#         'Alfa Solar Enerji': 'ALFAS.IS',
#         'Anadolu Sigorta': 'ANSGR.IS',
#         'Arçelik': 'ARCLK.IS',
#         'Aselsan': 'ASELS.IS',
#         'Astor Enerji': 'ASTOR.IS',
#         'Bera Holding': 'BERA.IS',
#         'Bosch Fren': 'BFREN.IS',
#         'Bien Yapı': 'BIENY.IS',
#         'Bim Birleşik Mağazalar': 'BIMAS.IS',
#         'Biotrend Çevre ve Enerji': 'BIOEN.IS',
#         'Borsa İstanbul': 'BOBET.IS',
#         'Borusan Yatırım': 'BRSAN.IS',
#         'Brisa Bridgestone Sabancı': 'BRYAT.IS',
#         'Batıçim': 'BTCIM.IS',
#         'Çanakkale Çimento': 'CANTE.IS',
#         'Coca Cola İçecek': 'CCOLA.IS',
#         'Çimsa Çimento': 'CIMSA.IS',
#         'CW Enerji': 'CWENE.IS',
#         'Doğuş Otomotiv': 'DOAS.IS',
#         'Doğan Holding': 'DOHOL.IS',
#         'Eczacıbaşı İlaç': 'ECILC.IS',
#         'Eczacıbaşı Yatırım': 'ECZYT.IS',
#         'Ege Endüstri': 'EGEEN.IS',
#         'Emlak Konut': 'EKGYO.IS',
#         'Enerjisa Enerji': 'ENERY.IS',
#         'Enka İnşaat': 'ENJSA.IS',
#         'Ereğli Demir ve Çelik': 'EREGL.IS',
#         'EuroPower Enerji': 'EUPWR.IS',
#         'Euro Yatırım Holding': 'EUREN.IS',
#         'Ford Otosan': 'FROTO.IS',
#         'Garanti Bankası': 'GARAN.IS',
#         'Girişim Elektrik': 'GESAN.IS',
#         'Gübre Fabrikaları': 'GUBRF.IS',
#         'Galata Wind': 'GWIND.IS',
#         'Halkbank': 'HALKB.IS',
#         'Hektaş': 'HEKTS.IS',
#         'İpek Doğal Enerji': 'IPEKE.IS',
#         'İş Bankası': 'ISCTR.IS',
#         'İş GYO': 'ISGYO.IS',
#         'İş Yatırım': 'ISMEN.IS',
#         'İzmir Demir Çelik': 'IZENR.IS',
#         'Kayseri Şeker': 'KAYSE.IS',
#         'Kardemir Karabük': 'KCAER.IS',
#         'Koç Holding': 'KCHOL.IS',
#         'Klimasan': 'KLSER.IS',
#         'Kontrolmatik Teknoloji': 'KONTR.IS',
#         'Konya Çimento': 'KONYA.IS',
#         'Koza Anadolu': 'KOZAA.IS',
#         'Koza Altın': 'KOZAL.IS',
#         'Kardemir': 'KRDMD.IS',
#         'Mavi Giyim': 'MAVI.IS',
#         'Migros': 'MGROS.IS',
#         'Mia Teknoloji': 'MIATK.IS',
#         'Odaş Elektrik': 'ODAS.IS',
#         'Otokar': 'OTKAR.IS',
#         'Oyak Çimento': 'OYAKC.IS',
#         'Petkim': 'PETKM.IS',
#         'Pegasus': 'PGSUS.IS',
#         'Quagr': 'QUAGR.IS',
#         'Reeder Teknoloji': 'REEDR.IS',
#         'Sabancı Holding': 'SAHOL.IS',
#         'Sasa Polyester': 'SASA.IS',
#         'Say Yenilenebilir Enerji': 'SAYAS.IS',
#         'SDT Uzay ve Savunma': 'SDTTR.IS',
#         'Şişecam': 'SISE.IS',
#         'Şekerbank': 'SKBNK.IS',
#         'Smart Güneş Enerjisi': 'SMRTG.IS',
#         'Şok Marketler': 'SOKM.IS',
#         'Tab Gıda': 'TABGD.IS',
#         'TAV Havalimanları': 'TAVHL.IS',
#         'Turkcell': 'TCELL.IS',
#         'Türk Hava Yolları': 'THYAO.IS',
#         'Tekfen': 'TKFEN.IS',
#         'Tofaş': 'TOASO.IS',
#         'TSKB': 'TSKB.IS',
#         'Türk Telekom': 'TTKOM.IS',
#         'Türk Traktör': 'TTRAK.IS',
#         'Tukaş': 'TUKAS.IS',
#         'Tüpraş': 'TUPRS.IS',
#         'Türk Sigorta': 'TURSG.IS',
#         'Ülker': 'ULKER.IS',
#         'VakıfBank': 'VAKBN.IS',
#         'Vestel Beyaz Eşya': 'VESBE.IS',
#         'Vestel': 'VESTL.IS',
#         'Yeo Teknoloji': 'YEOTK.IS',
#         'Yapı Kredi': 'YKBNK.IS',
#         'Yayla Agro': 'YYLGD.IS',
#         'Zorlu Enerji': 'ZOREN.IS'
#     }
#     borsasec = senetler[borsasec]
#     secim = borsasec

zaralik = range(1, 4500)
bugun = datetime.today()
prophet = st.sidebar.checkbox('Borsa Tahmini')
slider = st.sidebar.select_slider('Zaman Aralığı', options=zaralik, value=30)
aralik = timedelta(days=slider)
start = st.sidebar.date_input('Başlangıç Tarihi', value=bugun-aralik)   
end = st.sidebar.date_input('Bitiş Tarihi', value=bugun)
data = yf.download(secim, start=start, end=end)

def get_data(secim):
    data = yf.download(secim, start=start, end=end)
    st.line_chart(data['Close'])
    st.line_chart(data['Volume'])
    st.write(data.head())

    if prophet:
        data['ds'] = data.index
        data['y'] = data['Close']
        model = Prophet()
        model.fit(data)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        fig1 = model.plot(forecast)
        st.pyplot(fig1)
        fig2 = model.plot_components(forecast)
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
        # st.table(rsi)

    if fisl:
        macd = MACD(data)
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
        # st.table(hns)

# Chat-related code integration starts here
st.header('Stock Recommendation System')
openai_api_key = st.sidebar.text_input('OpenAI API Key',"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")





if openai_api_key:
    llm = ChatOpenAI(temperature=0, model_name='gpt-4-turbo', openai_api_key=openai_api_key)

    def get_stock_price(ticker):
        if "." in ticker:
            ticker = ticker.split(".")[0]
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        df = df[["Close","Volume"]]
        df.index = [str(x).split()[0] for x in list(df.index)]
        df.index.rename("Date", inplace=True)
        return df.to_string()

    def google_query(search_term):
        if "news" not in search_term:
            search_term = search_term + " stock news"
        url = f"https://www.google.com/search?q={search_term}"
        url = re.sub(r"\s", "+", url)
        return url

    def get_recent_stock_news(company_name):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
        g_query = google_query(company_name)
        res = requests.get(g_query, headers=headers).text
        soup = BeautifulSoup(res, "html.parser")
        news = []
        for n in soup.find_all("div", "n0jPhd ynAwRc tNxQIb nDgy9d"):
            news.append(n.text)
        for n in soup.find_all("div", "IJl0Z"):
            news.append(n.text)

        if len(news) > 6:
            news = news[:4]
        else:
            news = news
        
        news_string = ""
        for i, n in enumerate(news):
            news_string += f"{i}. {n}\n"
        top5_news = "Recent News:\n\n" + news_string
        
        return top5_news

    def get_financial_statements(ticker):
        if "." in ticker:
            ticker = ticker.split(".")[0]
        company = yf.Ticker(ticker)
        balance_sheet = company.balance_sheet
        if balance_sheet.shape[1] > 3:
            balance_sheet = balance_sheet.iloc[:, :3]
        balance_sheet = balance_sheet.dropna(how="any")
        balance_sheet = balance_sheet.to_string()
        return balance_sheet

    search = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name="Stock Ticker Search",
            func=search.run,
            description="Use only when you need to get stock ticker from internet, you can also get recent stock related news. Dont use it for any other analysis or task"
        ),
        Tool(
            name="Get Stock Historical Price",
            func=get_stock_price,
            description="Use when you are asked to evaluate or analyze a stock. This will output historic share price data. You should input the stock ticker to it"
        ),
        Tool(
            name="Get Recent News",
            func=get_recent_stock_news,
            description="Use this to fetch recent news about stocks"
        ),
        Tool(
            name="Get Financial Statements",
            func=get_financial_statements,
            description="Use this to get financial statement of the company. With the help of this data company's historic performance can be evaluated. You should input stock ticker to it"
        )
    ]

    zero_shot_agent = initialize_agent(
        llm=llm,
        agent="zero-shot-react-description",
        tools=tools,
        verbose=True,
        max_iteration=4,
        return_intermediate_steps=False,
        handle_parsing_errors=True
    )

    stock_prompt = """You are a financial advisor. Give stock recommendations for given query.
    Everytime first you should identify the company name and get the stock ticker symbol for the stock.
    Answer the following questions as best you can. You have access to the following tools:

    Get Stock Historical Price: Use when you are asked to evaluate or analyze a stock. This will output historic share price data. You should input the stock ticker to it 
    Stock Ticker Search: Use only when you need to get stock ticker from internet, you can also get recent stock related news. Dont use it for any other analysis or task
    Get Recent News: Use this to fetch recent news about stocks
    Get Financial Statements: Use this to get financial statement of the company. With the help of this data company's historic performance can be evaluaated. You should input stock ticker to it

    steps- 
    Note- if you fail in satisfying any of the step below, Just move to next one
    1) Get the company name and search for the "company name + stock ticker" on internet. Dont hallucinate extract stock ticker as it is from the text. Output- stock ticker. If stock ticker is not found, stop the process and output this text: This stock does not exist
    2) Use "Get Stock Historical Price" tool to gather stock info. Output- Stock data
    3) Get company's historic financial data using "Get Financial Statements". Output- Financial statement
    4) Use this "Get Recent News" tool to search for latest stock related news. Output- Stock news
    5) Analyze the stock based on gathered data and give detailed analysis for investment choice. provide numbers and reasons to justify your answer. Output- Give a single answer if the user should buy,hold or sell. You should Start the answer with Either Buy, Hold, or Sell in Bold after that Justify.

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do, Also try to follow steps mentioned above
    Action: the action to take, should be one of [Get Stock Historical Price, Stock Ticker Search, Get Recent News, Get Financial Statements]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times, if Thought is empty go to the next Thought and skip Action/Action Input and Observation)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""

    zero_shot_agent.agent.llm_chain.prompt.template = stock_prompt

    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = zero_shot_agent(f'Is {prompt} a good investment choice right now?', callbacks=[st_callback])
            st.write(response["output"])
