
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from keras import optimizers
from keras.constraints import maxnorm

# set fixed random seed for reproducibility
from numpy.random import seed
seed(56)
from tensorflow import set_random_seed
set_random_seed(56)

# load in all features
data = pd.read_csv('all_features_bitcoin', index_col='Unnamed: 0' )
data.columns

# from all features select subsets (according to appropriate feature ablation approach. This project found good results using 
# only public awareness features!)
x = data[[ 'AvgTone','group_size']]
y= data['rise_fall']

# convert to arrays and double check for any missing data
y = np.asarray(y)
df = np.asarray(x)
dt= df.astype(float)
np.where(np.isnan(dt))

# appropriate train test split
split = int(len(x)*0.85)
X_train, X_test, y_train, y_test = dt[:split], dt[split:], y[:split], y[split:]

# scaler X data
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Bayesian approximation using Dropout model. training = True line ensures Dropout active in test phase
def classifier2(X_train, y_train, X_test, y_test):
    
    inputs = keras.Input(shape=(X_train.shape[1],))
    x = keras.layers.Dense(28, activation='tanh')(inputs)
    x = keras.layers.Dropout(0.2)(x, training=True)
    predictions = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['acc'])
    model.fit(X_train, y_train, batch_size = 1, epochs = 100, verbose= 1)
    return model.predict(X_test) 

# Run model 30 iterations
def run_classifier(X_train, y_train, X_test, y_test):
    optimizer = Adam(lr=0.001)
    lst = []
    for i in range(30):
        y_pred = []
        y_pred = classifier2(X_train, y_train, X_test, y_test)
        lst.append(y_pred)
    return lst

# Mean of 30 model results is final result. Std of results is measure of model confidence
arr = np.asarray(run_classifier(X_train, y_train, X_test, y_test))
arr = arr.reshape((30, 165))
result = np.mean(arr, axis = 0)
std = np.std(arr, axis= 0)

# Convert sigmoid function outputs into binary
def binary_score(pred):
    lst =[]
    for i in pred:
        if i >= 0.5:
            lst.append(1)
        else:
            lst.append(0)
    return lst 

# Function to order predictions based on model confidence and provide an accuracy where x is the number of results to look at
def certain_predictions(std, result, y_test, x):
    score = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    std = list(enumerate(std,0))
    sorted_std = sorted(std, key=lambda y:y[1])
    binary_result = binary_score(result)
    
    for c,v in sorted_std[:x]:
        #if c % 5 == 0:
        #print(v)
        if binary_result[c] == 1 and y_test[c] == 1:
            score += 1
            TP += 1
        elif binary_result[c] == 0 and y_test[c] == 0:
            score += 1
            TN += 1
        elif binary_result[c] == 1 and y_test[c] == 0:
            FP += 1
        elif binary_result[c] == 0 and y_test[c] == 1:
            FN += 1
    return (score/x)*100, TP, TN, FP, FN 


def majority(y_test):
    d = {}
    for v in y_test:
        if v not in d:
            d[v] = 1
        else:
            d[v] += 1
    return d

# Investment performance program
def investment_ap(std, y, bin_score, z):

    cash = 0
    investment = 1000
    for c, v in std[:z]:
        #print(c, y[c], y[c+1], bin_score[c])
        if c <= 163:
            if bin_score[c] == 1:
                cash += ((investment / y[c]) * y[c+1]) - investment 
            else:
                cash += -(((investment / y[c]) * y[c+1]) - investment )
    return cash

# investment performance of a purely random strategy over X test set
def random_profit(y):
    cash = 0
    investment = 1000
    for i in range(100000):
        array = np.random.randint(165, size= 50)
        for j in array:
            if j <= 163:
                cash += ((investment / y[j]) * y[j+1]) - investment
        array = []
    return cash / 100000

