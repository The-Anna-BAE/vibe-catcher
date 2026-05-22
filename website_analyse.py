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
# # 1. Проверяем, где мы находимся
# current = os.getcwd()
# print(f"1. Текущая папка: {current}")

# # 2. Проверяем, какой путь мы добавляем
# parent = os.path.abspath(os.path.join(current, ".."))
# print(f"2. Ожидаемый путь к корню: {parent}")

# # 3. Проверяем, видит ли система папку со вторым проектом
# target_folder = os.path.join(parent, "sentiment_analysis")
# target_folder2 = os.path.join(parent, "scraper")

# # 4. Проверяем наличие __init__.py (это критично для некоторых окружений)
# init_file = os.path.join(target_folder, "__init__.py")
# init_file2 = os.path.join(target_folder2, "__init__.py")


# # 5. Добавляем в пути и смотрим, что видит Python
# if parent not in sys.path:
#     sys.path.append(parent)

# try:
#     import sentiment_analysis
#     import scraper
#     print("5. ИМПОРТ ПРОШЕЛ УСПЕШНО!")
# except ImportError as e:
#     print(f"5. ОШИБКА ИМПОРТА: {e}")

morph=MorphAnalyzer()
nltk.download('stopwords')

st.set_page_config(page_title='Анализ текста', page_icon='🕵️')


st.title('О чем молчат отзывы?')
st.write(
    '''
сайт для определения **тональности отзывов**
    '''
)


# st.title("Мой первый сайт")
# age = st.slider("Выберите ", 0, 100, 25)
# st.write(f"Ваш возраст: {age}")

st.write("Данный ресурс поможет провести и вывести **краткую** аналитику деятельности компании, основываясь на отзывах.")

name = st.text_input("Название компании")
link = st.text_input("Где искать отзывы (введите ссылку)")
if st.button("Рассчитать и вывести аналитику"):
    # try:
    with st.spinner("Пожалуйста подождите, я работаю :D"):
        
        links = another_scrapper.get_links(name, url=link)
    if links:
        st.success(f'Было посещено {len(links)} филиала(ов) "{name}"')
        progress_bar = st.progress(0)
        # cntr = parser.collect_reviews(links,'web_for_analitics/reviews_file.txt')
        data=[]
        for i,link in enumerate(links):
            bad_reviews= another_scrapper.scrapper(url=link)
            
            texts = [review['text'] for review in bad_reviews]

            data.extend(texts)
            # data.append(review for review in another_scrapper.scrapper(url=link))
            change=min(int((i+1)/len(links)*100),100)
            progress_bar.progress(change)
        another_scrapper.put_reviews_to_file(data=data, path_to_file='web_for_analitics/reviews_file.txt')
        reviews_to_analyse, lines,lines_b4 = analysis.prepare_file('web_for_analitics/reviews_file.txt')

        # df,df_text,df_labels = analysis.get_df('sentiment_analysis/dataset.csv')
        # data_values,review_values = analysis.vectorize_data(df_text,reviews_to_analyse['reviews'])
        # my_model = joblib.load('sentiment_analysis/sentimental_analysis_model.joblib')
        st.write(f'Количество отзывов для анализа: {len(lines_b4)}')
        results = model(lines_b4)
        res_labels = [res['label'] for res in results]
        print(res_labels)
        print()
        # labels = tf.keras.utils.to_categorical(df_labels)
        # X_train,X_val,Y_train,Y_val, y_train,y_test = analysis.split_data(data_values,labels)
        # input_shape = data_values.shape[1]

        # my_model = analysis.build_model(input_shape = input_shape)
        # my_model.compile(optimizer=optimizers.RMSprop(1e-4), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy','f1_score','precision','recall'])

        # history = my_model.fit(X_train,Y_train, batch_size=575,epochs= 10, verbose=1, validation_data=(X_val,Y_val))
        # res=my_model.predict(review_values)

    
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


        # cntr_pos=0
        # most_common_words_positive=[]

        # for k,v in flattend_positive.items():
        #     if cntr_pos==10:
        #         break
        #     word = morph(k)[0]
        #     part_speech=word.tag.POS
        #     if part_speech not in ['PREP','ADVB','CONJ','PRCL','INTJ']:
        #         most_common_words_positive.append((word,v))
        #         cntr_pos+=1

        
        # cntr_neg=0
        # most_common_words_negative=[]
        
        # for k,v in flattend_negative.items():
        #     if cntr_neg==10:
        #         break
        #     parse_results = morph(k)
        #     if not parse_results:  # Проверка: если список пустой, пропускаем итерацию
        #         continue
                
        #     word = parse_results[0]
        #     if part_speech not in ['PREP','ADVB','CONJ','PRCL','INTJ']:
        #         most_common_words_negative.append((word,v))
        #         cntr_neg+=1
        
        # most_common_words_positive=dict(most_common_words_positive)
        # most_common_words_negative=dict(most_common_words_negative)
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
    # except Exception as error:
    #     st.error(f'Хьюстон, у нас ошибка {error}')


# web_for_analitics/website_analyse.py

# https://yandex.by/showcaptcha?cc=1&form-fb-hint=8.25&mt=8B9B2A6A89D350FE1971EEE63D237D38140A802D8370E09FA0276E647D70B691C53C10BE384A79A5C384ABFA70C8E16E6D11497BF1C285698CEAB1219E5799A142CB37E713D1A0B987FDD2AE23BFE2F7544CF369D826284748BBAB50AF61970D88A0F9A1E92FD7F960A1DAA1BC6AB3E95F5C49DE67E6E264BCD93C18552BD11BED86A758F4A91F23276590B963F81CAB479CEF624B80C5837AC2736463EEAACB38EEFE203752E06BEC9F96F15C5B85CA22534A994C5615B833544FFD4BA95B30AD3F7DA072D4DF8633F6E08507C10D50C9A4FC888BD85770DDDFF1F84F5E283EEFEB31F73A4C292C7BD92BCCCEA6AEC937&retpath=aHR0cHM6Ly95YW5kZXguYnkvbWFwcy9vcmcvem53ci8yMzE0Njc3ODMyOTIvcmV2aWV3cz8%2C_97bf8aaae67a8338dccffc7b2afeac21&t=2%252F1777386943%252F670a4c87e2412494458cd667bafffb5f&u=8597954208555230503&s=841f785fa47afb0d05211b64d2638ca6