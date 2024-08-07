import pandas as pd
from bs4 import BeautifulSoup
import requests
import streamlit as st
from deep_translator import GoogleTranslator
from transformers import pipeline

@st.cache_data
def load_data():
    url = "https://www.cnbce.com/piyasalar"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    post_cards = soup.find_all('div', class_='post-card-title')

    titles = []
    for post_card in post_cards:
        a_tag = post_card.find('a')
        if a_tag:
            text = a_tag.get_text().strip()
            titles.append(text)

    translated_titles = []
    for title in titles:
        translated_title = GoogleTranslator(source='auto', target='en').translate(title)
        translated_titles.append(translated_title)

    df = pd.DataFrame({'Turkish': titles, 'English': translated_titles})
    return df


st.header("Stock Sentiment Analysis")

# Veriler yükleniyor spinnerı
with st.spinner('Veriler yükleniyor...'):
    df = load_data()
    st.write(df)

sentiment_model = pipeline("sentiment-analysis", model="soleimanian/financial-roberta-large-sentiment")

def analyze_sentiments(translated_titles):
    sentiments = sentiment_model(translated_titles)
    return sentiments

if st.button('Stock Sentiment Analysis'):
    sentiments = analyze_sentiments(df['English'].tolist())
    df['Sentiment'] = [s['label'] for s in sentiments]
    st.write(df)
