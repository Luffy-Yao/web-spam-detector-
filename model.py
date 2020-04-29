#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:34:44 2020

@author: linxing
"""

import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from nltk.util import ngrams
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import logging
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import mmh3
import pickle
level=logging.INFO
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=level)
log = logging.getLogger(__name__)


def data_clean(data,target,col_1,col_2,col_3):
   ## generate random seeds for reproducibility
    np.random.seed(3)
    
   ## import data from two sources as dataframes
    data[col_1]=[extract_tokens(i) for i in data[col_1]]
    data[col_2]=[extract_host(i) for i in data[col_2]]
    data[col_2]=[i[::-1] for i in data[col_2]]
    data[col_3]=data[col_1]+data[col_2]
    
    ## filter out any null values in target variable
    bool_series = pd.notnull(data[target]) 
    data=data[bool_series]
    y_label= np.array(data['label'].values.astype(int))
    
    ## 70/30 training and testing split on data 
    X=np.array(eng_hash(data[col_3].values))
    X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size=0.3)
    
    ## transform training and testing set into vectors with pre-fixed dimensions
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[2])
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[2])
    
    return (X_train,X_test,y_train,y_test)


def extract_tokens(element):
    tokens=str(element).rsplit("/")
    return tokens


def extract_host(element):
    host=str(element).rsplit(".")
    return host


def eng_hash(data, vdim=1000):
    ## take 3 n-gram of the url and hash it into a vector of length 1000
    final = []
    for url in data:
        v = [0] * vdim
        new = list(ngrams(url, 3))
        for i in new:
            new_ = ''.join(i)
            idx = mmh3.hash(new_) % vdim
            v[idx] += 1
        final.append([np.array(v)])
    return final


data=pd.read_csv("dump.csv")

X_train,X_test,y_train,y_test=data_clean(data,'label','url','host','host_url')

class LossHistory(keras.callbacks.Callback):
    # a class will capture the training loss
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
def construct_model():
    model = Sequential()
 
    ## configure hidden layers with dropout rate=0.15 and batchnormalization to prevent overfitting issues
    model.add(Dense(128, input_dim=1000))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.15))
        
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.15))

    ## final output layer for binary classification problem
    model.add(Dense(1, activation='sigmoid'))
    
    ## loss function: crossentropy, optimization_procedure: stochastic gradient descent, metrics: accuracy
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model

log.info("Beginning training model")
loss = LossHistory()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
model=construct_model()
history=model.fit(X_train, y_train,
              epochs=100,
              batch_size=128,validation_data=(X_test, y_test), verbose=1, callbacks=[es])


# evaluate the model
_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

## predicted values of X_test
y_pred=model.predict_classes(X_test,verbose=0)

## performance metrics
precision=precision_score(y_test,y_pred,average='weighted')
print(precision)
recall=recall_score(y_test, y_pred, average='weighted')
print(recall)
f1=f1_score(y_test,y_pred,average='weighted')
print(f1)
c_o_k=cohen_kappa_score(y_test,y_pred)
print(c_o_k)
roc=roc_auc_score(y_test,y_pred,average='weighted')
print(roc)
conf=confusion_matrix(y_test,y_pred)
print(conf)

# save the train val loss figure 
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.savefig("myfig.png")

# save model to disk
pickle.dump(model,open('model.pkl','wb'))


