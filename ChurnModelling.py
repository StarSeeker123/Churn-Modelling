# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:14:26 2020

@author: SUCHITRA
"""
import numpy as np #To work with arrays

import matplotlib.pyplot as plt #To plot graphs

import  pandas as pd #To import and manage datasets

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:,13].values


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 

ct = ColumnTransformer([("Geography", OneHotEncoder(), [1])],    remainder = 'passthrough')
X = ct.fit_transform(X)

from sklearn.preprocessing import LabelEncoder
laybelencoder_X=LabelEncoder()
X[:,4]=laybelencoder_X.fit_transform(X[:,4])

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)

#Building the artificial network
import keras

from keras.models import Sequential
from keras.layers import Dense

#creating input layer and first hidden layer
classifier = Sequential()
classifier.add(Dense(6, input_shape=(12,),activation='relu',kernel_initializer='uniform'))

#creating multiple hidden layers
classifier.add(Dense(6, activation='relu',kernel_initializer='uniform'))


#Output layer
classifier.add(Dense(1, activation='sigmoid',kernel_initializer='uniform'))
   

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train,Y_train,batch_size=10, nb_epoch=100)

y_pred=classifier.predict(X_test)
y_pred=(y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
