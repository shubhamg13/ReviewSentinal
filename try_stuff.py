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
import re

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


for i in range(len(X)):
    temp = X[i]
    X[i] = preprocessing(temp)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


def get_all_words(d1,d2):
    allw=[]
    for line in d1:
        for word in line.split():
            allw.append(word)
    for line in d2:
        for word in line.split():
            allw.append(word)

    return set(allw)
    
vectorizer = TfidfVectorizer(max_features=2900)
#vectorizer.fit(get_all_words(train,test))
train_X=vectorizer.fit_transform(X_train)
test_X=vectorizer.transform(X_test)


MNB = MultinomialNB()

MNB.fit(train_X, y_train)


pred = MNB.predict(test_X)

print(accuracy_score(y_test,pred))

dF_review = pd.read_csv(r"C:\Users\nauri\source\repos\NLP_Scraping\Sentiment_prog\Review(Content).csv")
#cleantext = re.sub(cleaner, '', text)

Ftext = []
for i in range(dF_review.shape[0]):
    temp = dF_review['Review Content'][i]
    pred_data = preprocessing(temp)
    Ftext.append(pred_data)

Ftext=vectorizer.transform(Ftext)
pred_res = MNB.predict(Ftext)

print(pred_res)

sum = 0
for i in pred_res:
    sum = sum + i
rating = sum/len(pred_res)
rating = rating * 5
print ("Rating out of 5 is :- ", rating)