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


import tensorflow as tf
from keras import layers
from keras import models
from keras import optimizers
import keras
import sys

import nltk
nltk.download('stopwords')


stop_words=set(stopwords.words('russian'))

morph=MorphAnalyzer()
def prepare_file (file_name):
    
    lines=[]
    lines_b4=[]
    with open (file_name,'r',encoding='utf-8') as reviews:
        for review in reviews:
            lines_b4.append(review.replace('\n',''))
            review=re.findall(r"[а-яё]+",str(review).lower())
            lines.append(" ".join(morph.parse(word)[0].normal_form for word in review if word not in stop_words))
    reviews_to_analyse = pd.DataFrame(lines,columns=['reviews'])
    return reviews_to_analyse,lines,lines_b4
# print(lines)
# reviews_to_analyse, lines,lines_b4 = prepare_file('sentiment_analysis/reviews_to_analyse.txt')

# print(reviews_to_analyse)

def get_df(path_to_csv):
    df=pd.read_csv(path_to_csv)

    if ',' in df.columns:
        df= df.drop([','], axis=1)
    df= df.dropna(subset=['text'])
    df_text=df['text']
    df_labels=df['label']
    return df,df_text,df_labels

if __name__=='__main__': 
    df,df_text,df_labels= get_df('sentiment_analysis/sentiment_dataset.csv')
    

def clean_text(review):
    review=re.findall(r"[а-яё]+",str(review).lower())
    review_lemmatized=" ".join(morph.parse(word)[0].normal_form for word in review if word not in stop_words)
    return review_lemmatized

def apply_changes_df(df):
    tqdm.pandas()
    df['text']= df['text'].progress_apply(clean_text)
    df_text=df['text']
    return df['text'],df_text


def vectorize_data(df_text,reviews_to_analyse):
    df_text=df_text.astype(str)
    reviews_to_analyse=reviews_to_analyse.astype(str)
    tf_idf = TfidfVectorizer(preprocessor=str,lowercase=False)
    data_values = tf_idf.fit_transform(df_text)
    review_values = tf_idf.transform(reviews_to_analyse['reviews'])
    df.to_csv('dataset.csv')
    return data_values,review_values


# data_values,review_values = vectorize_data(df_text,reviews_to_analyse['reviews'])


# def split_data(data_values,df_labels,stratify=df_labels):
    x_train,x_test,y_train,y_test=train_test_split(data_values,
                                                labels,
                                                random_state=42,
                                                test_size=0.2,
                                                stratify=stratify)

    X_train,X_val,Y_train,Y_val=train_test_split(x_train,
                                                y_train,
                                                random_state=42,
                                                test_size=0.25,
                                                stratify=y_train)
    return X_train,X_val,Y_train,Y_val, y_train,y_test


# labels = tf.keras.utils.to_categorical(df_labels)
# X_train,X_val,Y_train,Y_val, y_train,y_test = split_data(data_values,labels)

def prepare_table(res,lines,lines_b4):
    classes = sorted(set(res))
    d1 = dict()
    d2=dict()

    for cls in classes:
        d1[cls] =[lines_b4[i] for i, c in enumerate(res) if c==cls]
        d2[cls] =[lines[i] for i, c in enumerate(res) if c==cls]
    df1 = pd.DataFrame({i: pd.Series(text) for i,text in d1.items()})
    df2 = pd.DataFrame({i: pd.Series(text) for i,text in d2.items()})
    return df1,df2

# sorted_review_table, sorted_lemmatized_review_table = prepare_table(lines=lines,lines_b4=lines_b4)

# flattend = sorted_lemmatized_review_table[1].str.split().explode().value_counts()
# flattend.head(10)

# input_shape = data_values.shape[1]


# def build_model(input_shape):
    my_model= models.Sequential()
    my_model.add(layers.Input(shape =(input_shape,)))

    my_model.add(layers.Dense(264,activation='relu'))
    my_model.add(layers.BatchNormalization())
    my_model.add(layers.Dropout(0.25))

    my_model.add(layers.Dense(132,activation='relu'))
    my_model.add(layers.BatchNormalization())
    my_model.add(layers.Dropout(0.5))

    my_model.add(layers.Dense(66,activation='relu'))
    my_model.add(layers.BatchNormalization())

    my_model.add(layers.Dense(33,activation='relu'))
    my_model.add(layers.BatchNormalization())

    my_model.add(layers.Dense(16,activation='relu'))
    my_model.add(layers.BatchNormalization())
    my_model.add(layers.Dropout(0.25))

    my_model.add(layers.Dense(3,activation='softmax'))

    print('ok')
    return my_model

# my_model = build_model(input_shape = input_shape)
# my_model.compile(optimizer=optimizers.RMSprop(1e-4), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy','f1_score','precision','recall'])

# history = my_model.fit(X_train,Y_train, batch_size=575,epochs= 10, verbose=1, validation_data=(X_val,Y_val))