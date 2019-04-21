import NLP_Scraping
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import WordNetLemmatizer
import string
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from contextlib import closing
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver import Firefox
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time
#import urllib2
import urllib.request,urllib.parse,urllib.error
import re
from bs4 import BeautifulSoup
import unicodedata
from sklearn.externals import joblib as jb



data = pd.read_csv(r"C:\Users\nauri\Desktop\ML proj\Amazon_Unlocked_Mobile.csv")
X = data['Reviews'][:]
Y = data['Rating'][:]

for i in range(len(Y)):
    temp = Y[i]
    temp = int(temp)
    if temp >= 3:
        temp = 1
    elif temp < 3:
        temp = 0
    Y[i] = temp


def preprocessing(text):
    
    # tokenize into words
    tokens = str(text).split()
    for i,word in enumerate(tokens):
        if (word in ['not','no'] or "n't" in word) and (i != len(tokens) - 1):
            tokens.append("not_" + tokens[i+1])
            del tokens[i]
            del tokens[i+1]

    # remove stopwords
    stop = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop]

    # remove words less than three letters
    tokens = [word for word in tokens if len(word) >= 3]

    # lower capitalization
    tokens = [word.lower() for word in tokens]

    # lemmatize
    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(word) for word in tokens]
    preprocessed_text= ' '.join(tokens)

    return preprocessed_text 

print("Processing text data...")
for i in range(len(X)):
    temp = X[i]
    X[i] = preprocessing(temp)

print("Splitting Dataset for training and testing...")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

    
vectorizer = TfidfVectorizer(max_features=2900)
train_X=vectorizer.fit_transform(X_train)
test_X=vectorizer.transform(X_test)


MNB = MultinomialNB()

print("Training classifier...")
MNB.fit(train_X, y_train)

jb.dump(MNB, 'pickled_model.pkl')
print("The model has been pickled for further testing...")

pred = MNB.predict(test_X)

print("The accuracy of our classifier is as follows : ", accuracy_score(y_test,pred))

dF_review = pd.read_csv(r"C:\Users\nauri\source\repos\NLP_Scraping\NLP_Scraping\Review(Content - Redmi Note 5 Pro).csv")
print("Predicting Comment data from scraping...")
Ftext = []
for i in range(dF_review.shape[0]):
    temp = dF_review['Review Content'][i]
    pred_data = preprocessing(temp)
    Ftext.append(pred_data)

camera_file = open(r"Camera words.txt", "r")
filter = camera_file.read().split('\n')

Ftext_camera = []
for l in Ftext:
    camera_text = l.split(".")
    for c in camera_text:
        for w in filter:
            if w in c and c not in Ftext_camera:
                Ftext_camera.append(c)

Ftext=vectorizer.transform(Ftext)
Ftext_camera=vectorizer.transform(Ftext_camera)

pred_res = MNB.predict(Ftext)
pred_camera = MNB.predict(Ftext_camera)
print("The Predictions for each comment are as follows :-")
print(pred_res)
print(len(pred_res))
print("The Predictions for all camera related comments are as follows :-")
print(len(pred_camera))

sum = 0
sum1 = 0
for i in pred_res:
    sum = sum + i
rating = sum/len(pred_res)
rating = rating * 5
print ("Rating out of 5 is :- ", rating)
for i in pred_camera:
    sum1 = sum1 + i
rating1 = sum1/len(pred_camera)
rating1 = rating1 * 5   
print ("Rating out of 5 for camera is :- ", rating1)

#total_neg = 0
#total_pos = 0
#for i in pred_res:
#    if i == 0:
#        total_neg = total_neg + 1
#    elif i == 1:
#        total_pos = total_pos + 1

#total_neg1 = 0
#total_pos1 = 0
#for i in pred_camera:
#    if i == 0:
#        total_neg1 = total_neg1 + 1
#    elif i == 1:
#        total_pos1 = total_pos1 + 1
#import matplotlib.pyplot as plt
#objects = ['Positive','Negative']
#y_pos = np.arange(len(objects))

#plt.bar(y_pos,[total_pos,total_neg],alpha=0.5)
#plt.xticks(y_pos,objects)
#plt.ylabel('Number')
#plt.title('Number of Postive and Negative Reviews')

#plt.show()