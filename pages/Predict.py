import yfinance as yf
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import streamlit as st

def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df

def model_engine(model, num):
    # yalnızca kapanış fiyatını alma
    df = data[['Close']]
    # kapanış fiyatı tahmin edilen gün sayısına göre değiştiriliyor
    df['preds'] = df.Close.shift(-num)
    # verileri ölçeklendirme
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # son gün sayısı verilerinin saklanması
    x_forecast = x[-num:]
    # tahmin için gerekli değerleri seçme
    x = x[:-num]
    # preds sütununu alma
    y = df.preds.values
    # tahmin için gerekli değerleri seçme
    y = y[:-num]

    #verileri ayırma
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # modeli tahmin etme
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.write(f'Doğruluk Oranı Tahmini : {r2_score(y_test, preds)}')
    # gün sayısına göre hisse senedi fiyatını tahmin etmek
    forecast_pred = model.predict(x_forecast)
    day = 10
    for i in forecast_pred:
        st.write(f'{day}. Gün İçin Tahmini Kapanış Fiyatı : {i}')
        day += 1

stock=st.text_input("Hisse Senedi Kodu Giriniz","AEFES.IS")
btn=st.button("Predict")
if btn:
    today = datetime.date.today()
    demo = datetime.datetime.strptime('2024-06-27', '%Y-%m-%d')
    duration = 3000
    before = demo - datetime.timedelta(days=duration)
    start_date = before

    end_date = demo-datetime.timedelta(days=0) 
    st.write(end_date)
    scaler = StandardScaler()

    data = download_data(stock,start_date,end_date)

    num = 1

    engine = LinearRegression()
    model_engine(engine, num)
