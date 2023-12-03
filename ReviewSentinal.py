import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

X = []
Y = []
Vectorizer = TfidfVectorizer(max_features=2900)
MNB = MultinomialNB()


def read_data():
    print("Reading Data")
    global X, Y
    data = pd.read_csv(r"./Amazon_Unlocked_Mobile.csv")
    X = data['Reviews'][:]
    Y = data['Rating'][:]
    for i in range(len(Y)):
        if Y[i] < 3:
            Y[i] = 0
        else:
            Y[i] = 1


def tokenize(text):
    # tokenize into words
    tokens = str(text).split()
    for i, word in enumerate(tokens):
        if (word in ['not', 'no'] or "n't" in word) and (i != len(tokens) - 1):
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
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def preprocess():
    print("Preprocessing Data")
    global X
    for i in range(len(X)):
        X[i] = tokenize(X[i])


def train_model():
    print("Splitting Dataset for training and testing...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=0)

    train_X = Vectorizer.fit_transform(X_train)
    test_X = Vectorizer.transform(X_test)

    print("Training classifier...")
    MNB.fit(train_X, Y_train)

    predicted = MNB.predict(test_X)
    print("The accuracy of our classifier is as follows : ",
          accuracy_score(Y_test, predicted))


def predict_result():
    reviews = pd.read_csv(r"./Review.csv")

    print("Predicting Comment data from scraping...")
    tokenized_text = []
    for i in range(reviews.shape[0]):
        temp = reviews['Review Content'][i]
        tokens = tokenize(temp)
        tokenized_text.append(tokens)

    camera_file = open(r"CameraWords.txt", "r", encoding='latin-1')
    filters = camera_file.read().split('\n')

    filtered_text = []
    for sentences in tokenized_text:
        sentence = sentences.split(".")
        for word in sentence:
            for filter in filters:
                if filter in word and word not in filtered_text:
                    filtered_text.append(word)

    tokenized_text = Vectorizer.transform(tokenized_text)
    filtered_text = Vectorizer.transform(filtered_text)

    pred_res = MNB.predict(tokenized_text)
    pred_camera = MNB.predict(filtered_text)

    print("Overall Prediction : ", np.sum(pred_res)/len(pred_res))
    print("Prediction as considering Camera : ",
          np.sum(pred_camera)/len(pred_camera))


read_data()
preprocess()
train_model()
predict_result()
