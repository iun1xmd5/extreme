#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 17:12:27 2020
@author: c1ph3r
"""

from numpy import array
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Flatten, LSTM, TimeDistributed,Input,RepeatVector, Dropout, Average,RepeatVector
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras import backend as K
from keras.utils import plot_model
from keras.layers.merge import concatenate
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import tqdm
from tweek import set_size
from bayes_opt import BayesianOptimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
plt.rc('font',**{'family':'serif','serif':['Times']})
plt.rc('text', usetex = True) # Use latex for text
plt.rc('xtick', labelsize = 10)
plt.rc('ytick', labelsize = 10)
plt.rc('axes', labelsize = 10)
plt.rc('axes', linewidth=1)
plt.rcParams['text.latex.preamble'] =[r'\boldmath']
params = {'legend.framealpha': 0.1,
          'legend.handlelength': 0.8,
          'legend.labelspacing':0.1,
          'legend.fontsize' : 10}
plt.rcParams.update(params)
class Ofat():
    def __init__(self,X,y,sr,q=0.05,batch_size=450,epochs=3):
        self.X = X
        self.y = y
        self.w=X.shape[2]
        self.scaler = StandardScaler()
        #self.X = self.scaler.fit_transform(X.reshape(-1, X.shape[1]))
        #self.y = self.scaler.fit_transform(y.reshape(-1,y.shape[0])).reshape(-1)
        self.q= q
        self.sr = sr
        self.sequence_length=X.shape[1]
        self.batch_size= batch_size
        self.epochs =epochs
        #self.X = self.scaler.fit_transform(self.X.reshape(-1,self.sequence_length)).reshape(-1,self.sequence_length,self.w)
        #self.X = self.X.reshape(-1,1,self.sequence_length,X.shape[2])
        self.X_train, self.y_train = self.X[:self.sr,:], self.y[:self.sr]

    def fit(self):
        #input channel 1
        inputs1 = Input(shape=(self.sequence_length, self.w))
        conv1= Conv1D(filters=32,
                                      kernel_size=3,
                                      activation='relu')(inputs1)
        drop1 =Dropout(0.3)(conv1)
        #conv2 = TimeDistributed(Conv1D(filters=16, kernel_size=1, activation='relu'))(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)
        #input channel 2
        inputs2 = Input(shape=(self.sequence_length, self.w))
        conv2= Conv1D(filters=32,
                                      kernel_size=3,
                                      activation='relu')(inputs2)
        drop2 =Dropout(0.3)(conv2)
        #conv2 = TimeDistributed(Conv1D(filters=16, kernel_size=1, activation='relu'))(conv1)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 =Flatten()(pool2)
        #input channel 3
        inputs3 = Input(shape=(self.sequence_length, self.w))
        conv3= Conv1D(filters=32,
                                      kernel_size=2,
                                      activation='relu')(inputs3)
        drop3 = Dropout(0.3)(conv3)
        #conv3 = TimeDistributed(Conv1D(filters=16, kernel_size=1, activation='relu'))(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)
        #merged
        merged =concatenate([flat1,flat2, flat3])
        rv = RepeatVector(self.w)(merged)
        lstm1= LSTM(128,
                    activation='relu',
                    return_sequences=True,
                    dropout=0.2)(rv,training=True)
        lstm2 =  LSTM(32,
                      activation='relu',
                      dropout=0.2)(lstm1, training=True)
        dense1 = Dense(50)(lstm2)
        out10 = Dense(1)(dense1)
        out50 = Dense(1)(dense1)
        out90 = Dense(1)(dense1)

        model = Model([inputs1, inputs2, inputs3], [out10,out50,out90])
        losses = [lambda y,f:self.loss(self.q, y, f),
                  lambda y,f:self.loss(0.5, y, f),
                  lambda y,f: self.loss(1-self.q, y, f)]
        #optimizer=BayesianOptimization()
        model.compile(loss=losses, optimizer='adam', metrics=['mae'],loss_weights = [0.2,0.2,0.2])
        model.fit([self.X_train,self.X_train,self.X_train],
                            [self.y_train, self.y_train, self.y_train],
                            epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)
        #self.model = model
        #self.history=history

    def predict(self,X,y,mc=5):
        #X = self.scaler.fit_transform(X.reshape(-1, X.shape[1])) #for avo
        #X=X.reshape(-1,self.sequence_length,self.w)
        self.x_test, self.y_test= X,y
        self.ypred=np.array(self.model.predict([self.x_test,
                                                self.x_test,
                                                self.x_test], verbose=1)).reshape(3,-1)
        self.y10, self.y50, self.y90 = self.ypred[0], self.ypred[1], self.ypred[2]
        y_pred10, y_pred50, y_pred90 = [], [], []
        for i in tqdm.tqdm(range(0,mc)):
            pred = self.predictor()
            y_pred10.append(pred[0])
            y_pred50.append(pred[1])
            y_pred90.append(pred[2])
        self.ypred10 = np.mean(np.hstack(np.asarray(y_pred10)),axis=1).reshape(-1)
        self.ypred50 = np.mean(np.hstack(np.asarray(y_pred50)),axis=1).reshape(-1)
        self.ypred90 = np.mean(np.hstack(np.asarray(y_pred90)),axis=1).reshape(-1)

        self.error= self.y_test-self.y50
        self.smae_score = abs(self.y_test-self.y50)/(abs(self.y_test)+abs(self.y50))
        self.smae=self.smae_score.mean()
        self.mean_s, self.std_s= np.mean(abs(self.smae_score)), np.std(self.smae_score)
        print('\n mc= %d smae=%0.4f, mean=%0.4f, std=%0.4f'
              %(mc,self.smae, self.mean_s, self.std_s),end='\n')

    def predictor(self):
        #xtest=xtest.reshape(-1,1,self.sequence_length,self.w)
        #X_test = self.scaler.transform(xtest.reshape(-1,self.sequence_length)).reshape(-1,1,self.sequence_length,1)
        NN = K.function([self.model.layers[2].input, self.model.layers[1].input, self.model.layers[0].input,
                         K.learning_phase()],
                        [self.model.layers[-3].output, self.model.layers[-2].output, self.model.layers[-1].output])
        #trans_pred = self.scaler.transform(X_test.reshape(-1,self.sequence_length)).reshape(-1,1,self.sequence_length,1)
        #xt=[xtest,xtest,xtest]
        NN_pred = np.array(NN([[self.x_test,self.x_test,self.x_test],1]))
        return NN_pred

    def _predict(self, name='wiki1'):
        fraction = 1
        width =212
        fig, ax = plt.subplots(figsize=set_size(width, fraction), sharex='all', gridspec_kw={'hspace': 0.5})
        #plt.plot(self.y10, color='green', alpha=0.9)
        plt.plot(self.y50, color='red', alpha=0.8, label='OFAT')
        plt.plot(self.y_test, color='olive', alpha=0.8, label ='ground truth')
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
        ax.xaxis.set_major_locator(plt.LinearLocator(6))
        ax.yaxis.set_major_locator(plt.LinearLocator(6))
        plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.05), ncol=2)
        plt.xlabel('Time (days)')
        #print('{} smae: {} std: {}'.format(name, self.smae, self.std_s))
        plt.ylabel('Weekly prices')
        plt.tight_layout()
        # plt.savefig('../../ofat/figures/'+
        #                 name+'.pdf',format='pdf',bbox_inches='tight')

    def plot(self,name='uncertainty'):
        fraction = 0.5
        width =510
        fig, ax = plt.subplots(figsize=set_size(width, fraction), sharex='all')
        plt.plot(self.y50,'red')
        plt.plot(self.y_test,'orange')
        ax.fill_between(range(0,len(self.y50)), self.y10,self.y90,color='grey',alpha=0.5)
        #ax.grid(True, zorder=5)
        plt.plot(self.y50, color='red', alpha=0.8, label='OFAT')
        plt.plot(self.y_test, color='olive', alpha=0.8, label ='ground truth')
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
        ax.xaxis.set_major_locator(plt.LinearLocator(6))
        ax.yaxis.set_major_locator(plt.LinearLocator(6))
        #plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.05), ncol=2)
        ax.set_xlabel('Time (weeks)')
        ax.set_ylabel('Average price')
        plt.tight_layout()
        # plt.savefig('../_8th/papers/vldb/figures/'+
        #              name+'.pdf',format='pdf',bbox_inches='tight')

    def loss(self,q,y,f):
        e = y-f
        return K.mean(K.maximum(q*e, (q-1)*e),axis=-1)












