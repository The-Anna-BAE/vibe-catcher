import sys
import os

import streamlit as st
import another_scrapper
import analysis


import re
from nltk.corpus import stopwords
# from pymystem3 import Mystem
import pandas as pd
from tqdm import tqdm
from pymorphy3 import MorphAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

import streamlit as st
import plotly.express as px


import tensorflow as tf
from keras import layers
from keras import models
from keras import optimizers
import keras

from transformers import pipeline

import nltk

import asyncio


model_name = "cointegrated/rubert-tiny-sentiment-balanced"
model = pipeline("sentiment-analysis", model=model_name)


labels = ['positive','negative','neutral']




if sys.platform == 'win32':
    if not isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
morph=MorphAnalyzer()
nltk.download('stopwords')

st.set_page_config(page_title='Анализ текста', page_icon='🕵️')


st.title('О чем молчат отзывы?')
st.write(
    '''
сайт для определения **тональности отзывов**
    '''
)

st.write("Данный ресурс поможет провести и вывести **краткую** аналитику деятельности компании, основываясь на отзывах.")

name = st.text_input("Название компании")
link = st.text_input("Где искать отзывы (введите ссылку)")
if st.button("Рассчитать и вывести аналитику"):
    try:
        with st.spinner("Пожалуйста подождите, я работаю :D"):
            
            links = another_scrapper.get_links(name, url=link)
        if links:
            st.success(f'Было посещено {len(links)} филиала(ов) "{name}"')
            progress_bar = st.progress(0)
            data=[]
            for i,link in enumerate(links):
                bad_reviews= another_scrapper.scrapper(url=link)
                
                texts = [review['text'] for review in bad_reviews]

                data.extend(texts)

                change=min(int((i+1)/len(links)*100),100)
                progress_bar.progress(change)
            another_scrapper.put_reviews_to_file(data=data, path_to_file='web_for_analitics/reviews_file.txt')
            reviews_to_analyse, lines,lines_b4 = analysis.prepare_file('web_for_analitics/reviews_file.txt')

            st.write(f'Количество отзывов для анализа: {len(lines_b4)}')
            results = model(lines_b4)
            res_labels = [res['label'] for res in results]
            
            sorted_review_table, sorted_lemmatized_review_table = analysis.prepare_table(lines=lines,lines_b4=lines_b4,res=res_labels)


            sorted_review_table.columns=['Негативные','Нейтральные','Позитивные']
            sorted_lemmatized_review_table.columns=['Негативные','Нейтральные','Позитивные']
            
            sorted_review_table.to_csv(index=False, path_or_buf='result.csv')
            
            flattend_positive = sorted_lemmatized_review_table['Позитивные'].str.split().explode().value_counts()
            most_common_words_positive=flattend_positive.head(10)

            flattend_negative = sorted_lemmatized_review_table['Негативные'].str.split().explode().value_counts()
            most_common_words_negative=flattend_negative.head(10)

            counts = sorted_review_table.count()
            counts = counts.reset_index()
            counts.columns = ['Оценка отзыва', 'Количество']

            st.subheader("Результат аналитики")


            st.dataframe(counts
        ,hide_index=True
            )

            st.subheader("10 наиболее часто упоминаемых слова среди")
            col1,col2 = st.columns(2)
            
            with col1:
                st.subheader('позитивных отзывов')
                df1 = pd.DataFrame(most_common_words_positive.items(), columns=['слово', 'количество упоминаний'])
                st.dataframe(df1,hide_index=True)
            with col2:
                st.subheader('негативных отзывов')
                df2 = pd.DataFrame(most_common_words_negative.items(), columns=['слово', 'количество упоминаний'])
                st.dataframe(df2,hide_index=True)
            fig = px.pie(counts,
                            names='Оценка отзыва',
                            values= 'Количество',
                            title="Результат в виде круговой диаграммы",
            color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(textinfo='percent+value')


            csv = sorted_review_table.to_csv(index=False, na_rep='нет текста').encode('utf-8')

            st.plotly_chart(fig, width="stretch")
            
            st.download_button(label='Скачать таблицу', data= csv, file_name='table.csv',mime='text/csv')
        else:
            st.error('Что-то пошло не так')
    except Exception as error:
        st.error(f'Хьюстон, у нас ошибка {error}')


# web_for_analitics/website_analyse.py

# https://yandex.by/showcaptcha?cc=1&form-fb-hint=8.25&mt=8B9B2A6A89D350FE1971EEE63D237D38140A802D8370E09FA0276E647D70B691C53C10BE384A79A5C384ABFA70C8E16E6D11497BF1C285698CEAB1219E5799A142CB37E713D1A0B987FDD2AE23BFE2F7544CF369D826284748BBAB50AF61970D88A0F9A1E92FD7F960A1DAA1BC6AB3E95F5C49DE67E6E264BCD93C18552BD11BED86A758F4A91F23276590B963F81CAB479CEF624B80C5837AC2736463EEAACB38EEFE203752E06BEC9F96F15C5B85CA22534A994C5615B833544FFD4BA95B30AD3F7DA072D4DF8633F6E08507C10D50C9A4FC888BD85770DDDFF1F84F5E283EEFEB31F73A4C292C7BD92BCCCEA6AEC937&retpath=aHR0cHM6Ly95YW5kZXguYnkvbWFwcy9vcmcvem53ci8yMzE0Njc3ODMyOTIvcmV2aWV3cz8%2C_97bf8aaae67a8338dccffc7b2afeac21&t=2%252F1777386943%252F670a4c87e2412494458cd667bafffb5f&u=8597954208555230503&s=841f785fa47afb0d05211b64d2638ca6