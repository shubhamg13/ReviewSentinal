import pandas as pd

X = []
Y = []

def read_data():
    print("Reading Data")
    global X, Y
    data = pd.read_csv("./Amazon_Unlocked_Mobile.csv")
    X = data['Reviews'][:]
    Y = data['Rating'][:]
    for i in range(len(Y)):
        if Y[i] < 3:
            Y[i] = 0
        else:
            Y[i] = 1
            
read_data()            
print(Y)