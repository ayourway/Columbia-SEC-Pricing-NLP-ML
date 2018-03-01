#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 22:50:31 2018

@author: Jun Guo
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn import cross_validation
from sklearn.model_selection import cross_val_score, GridSearchCV

from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, Dense, Dropout, Reshape, BatchNormalization, TimeDistributed, Lambda, Activation, LSTM, Flatten, Convolution1D, RepeatVector
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard
#from keras import initializers
from keras import backend as K
from keras.optimizers import SGD
from keras.optimizers import Adadelta
from keras.preprocessing import sequence
from keras import optimizers
from keras.utils import np_utils


def mask_last(x):
    """
    https://stackoverflow.com/a/31226290/7836408
    """    
    result = np.ones_like(x)
    result[-2:] = 0
    return result


def split_xy(table):
    X = table.loc[:,'Negative':'WCount'].values
    y = table.result.values
    return X, y


def random_forest(train_X, test_X, train_y, test_y):
    rnd = RandomForestClassifier(n_estimators = 6)
    rnd.fit(train_X, train_y)

    y_predict = rnd.predict(test_X)
    print("Accuracy: {}".format(accuracy_score(test_y, y_predict)))
    print("F1 Score: {}".format(f1_score(test_y, y_predict)))
    print("Feature Ranking: {}".format(\
          sentiments[np.argsort(rnd.feature_importances_)[::-1]]))
    return y_predict


def GBclassifer(train_X,test_X,train_y,test_y):
    predictions_list = []
    for i in range(1,50):
        model = GradientBoostingClassifier(n_estimators=i,max_depth=2)
        model.fit(train_X, train_y)
        y_predict = model.predict(test_X)
        predictions_list.append(accuracy_score(test_y, y_predict))
    return max(predictions_list), predictions_list.index(max(predictions_list)) + 1


def KNN(train_X,test_X,train_y,test_y):
    predictions_list = []
    for i in range(1,25):
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(train_X, train_y)
        y_predict = neigh.predict(test_X)
        predictions_list.append(accuracy_score(test_y, y_predict))
    return max(predictions_list), predictions_list.index(max(predictions_list)) + 1


def create_UniLSTM(train_X, train_y, hidden_size, num_layers):
    # define model
    model = Sequential()
    model.add(LSTM(hidden_size, return_sequences=True, 
                   input_shape = (train_X.shape[1], train_X.shape[2])))
    for _ in range(num_layers):
        model.add(LSTM(hidden_size, return_sequences=True))
        model.add(Activation("sigmoid"))

    model.add(TimeDistributed(Dense(2, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    
    return model


def split_traintest(table):
    Tic = table.Tic.unique()
    idx = [1]* int(len(Tic) * 0.8) + [0] * (len(Tic) - int(len(Tic) * 0.8))
    np.random.shuffle(idx)
    
    train_idx = Tic[np.array(idx).astype(bool)]
    test_idx = Tic[np.abs(np.array(idx)-1).astype(bool)]
    
    temp = table.set_index('Tic', drop = False)
    return temp.loc[train_idx], temp.loc[test_idx], train_idx, test_idx
    
def split_xy_timestep(table, table_idx, maxlen):
    X_timestep = []
    y_timestep = []
    for tic in table_idx:
        X, y = split_xy(table.loc[tic:tic])
        if len(X) >= maxlen:
            Xpad = X[:maxlen]
            ypad = np_utils.to_categorical(y[:maxlen], num_classes= 2)
        else: 
            pass
#            pad = maxlen - len(X)
#            Xpad = np.vstack((np.zeros((pad, X.shape[1])), X))
#            ypad = np_utils.to_categorical(np.hstack((np.zeros(pad), y)))
            
        X_timestep.append(np.expand_dims(Xpad, axis = 0))
        y_timestep.append(np.expand_dims(ypad, axis = 0))
    X_ts = np.vstack(X_timestep)
    y_ts = np.vstack(y_timestep)
    
    return X_ts, y_ts

def fully_connected(train_X, train_y, hidden_size, num_layers):
    model = Sequential()
    model.add(Dense(hidden_size, activation='relu', input_shape= train_X.shape[1:]))
    for _ in range(num_layers):
        model.add(Dense(hidden_size, activation='relu'))
        model.add(Dropout(0.2))
    
    model.add(Dense(trainy.shape[-1], activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    model.summary()
    return model
    
combined_matrix = pd.read_csv("../data/combined_matrix.csv")

sentiments = combined_matrix.keys()[3:-1]

train, test, train_idx, test_idx = split_traintest(combined_matrix)

trainX, trainy = split_xy_timestep(train, train_idx, 48)
testX, testy = split_xy_timestep(test, test_idx, 48)
#trainX, trainy = split_xy(train)
#testX, testy = split_xy(test)
#trainy = np_utils.to_categorical(trainy, num_classes= 2)
#testy = np_utils.to_categorical(testy, num_classes= 2)

model = create_UniLSTM(trainX, trainy, 128, 3)
model.fit(trainX, trainy, batch_size= 50, epochs = 40)
scores = model.evaluate(testX, testy, verbose=1)
print(scores)
#y_forest = random_forest(trainX, testX, np.argmax(trainy, axis =1), np.argmax(testy, axis =1))
#
##print(GBclassifer(trainX, testX, trainy, testy))
#print(1KNN(trainX, testX, trainy, testy))
#
#if __name__ == '__main__':
