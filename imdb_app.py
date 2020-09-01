# -*- coding: utf-8 -*-
"""IMDB-App.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sHsQ95X4B9E3FEIXjr0D9-VS_33Dpxpv
"""

# Commented out IPython magic to ensure Python compatibility.
# %%writefile maj-imdb1.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# 
# st.title(" Machine Learning Major Project ")
# st.subheader(" IMDB Review DataSet ")
# st.write(" Classifier: NaiveBayes")
# st.write(" Accuracy: 0.87 ")
# 
# data = pd.read_csv('/content/drive/My Drive/Python SmartKnower/Machine Learning/IMDB Dataset.csv',usecols=['review','sentiment'])
# 
# x1 = data.iloc[:,0].values
# y1  =data['sentiment'].values
# 
# sentence = st.text_input(" Write your review here : ")
# 
# model1 = Pipeline([('tfidf',TfidfVectorizer()),('model',MultinomialNB())])
# model1.fit(x1,y1)
# 
# if sentence:
#   y_pred = model1.predict([sentence])
#   st.write(" Prediction:")
#   st.write(y_pred)

