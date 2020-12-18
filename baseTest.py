#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 05:00:11 2020
@author: c1ph3r
"""
from memory_profiler import profile
#import time
from keras.models import Model, Input
from keras.layers import LSTM,Dense,Conv1D,MaxPooling1D,Flatten,Dropout,concatenate,RepeatVector,TimeDistributed, BatchNormalization,Bidirectional, ConvLSTM2D
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from statsmodels.tsa.arima_model import ARIMA
#from ofat import Ofat
#from fbprophet import Prophet
from tweek import set_size
import os
#from pyramid.arima import auto_arima
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from statsmodels.tsa.statespace.sarimax import SARIMAX
#import itertools
#import statsmodels.api as sm
#from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error
import keras.backend as K
from sklearn.preprocessing import StandardScaler

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

class Evaluation():
    def __init__(self,
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 epochs=3,
                 q=0.05,
                 batch_size=128,
                 verbose=1):
        self.verbose=verbose
        self.epochs=epochs
        self.batch_size=batch_size
        self.X_train=X_train
        self.y_train=y_train
        self.q=q
        self.epochs=epochs
        self.X_test, self.y_test=X_test,y_test
        self.scalar= StandardScaler()
        self.sequence_length=X_train.shape[1]
        self.w=X_train.shape[2]


    def qlstm(self):
        print('\n Entering into QLSTM')
        start = time.time()
        input1 = Input(shape=(self.X_train.shape[1],self.X_train.shape[2]))
        lstm1= Bidirectional(LSTM(128,
                    activation='relu',
                    return_sequences=True,
                    dropout=0.2))(input1,training=True)
        lstm2 =  Bidirectional(LSTM(32,
                      activation='relu',
                      dropout=0.2))(lstm1, training=True)
        dense1 = Dense(50)(lstm2)
        out10 = Dense(1)(dense1)
        out50 = Dense(1)(dense1)
        out90 = Dense(1)(dense1)
        model = Model(inputs=input1, outputs=[out10, out50, out90])
        losses = [lambda y,f:self.loss(self.q, y, f),
                  lambda y,f:self.loss(0.5, y, f),
                  lambda y,f: self.loss(1-self.q, y, f)]

        monitor = EarlyStopping(monitor='loss',min_delta=3e-3, patience=20,verbose=1)
        model.compile(loss=losses, optimizer='adam', metrics=['mae'],loss_weights = [0.2,0.2,0.2])
        history = model.fit([self.X_train],
                            [self.y_train, self.y_train, self.y_train],
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            verbose=self.verbose, shuffle=True,
                            callbacks=[monitor])
        yhat = np.array(model.predict(self.X_test,verbose=0)).reshape(3,-1)
        self.yqlstm=yhat
        self.qlstmt = time.time()-start
        self.modelql, self.historyql = model, history
        print('Execution time: {}'.format(self.qlstmt), end='\n')

    def lstmnet(self):
        print('Entering into LSTM net......', end='\n')
        start = time.time()
        input1 = Input(shape=(self.X_train.shape[1],self.X_train.shape[2]))
        lstm1 = LSTM(128,
                     activation='relu',
                     return_sequences=True)(input1, training=True)
        lstm2 = LSTM(32,
                     activation='relu')(lstm1,training=True)
        dense1 = Dense(50)(lstm2)
        output = Dense(1)(dense1)
        model =Model(inputs=input1,outputs=output)
        monitor = EarlyStopping(monitor='loss',min_delta=3e-3, patience=20,verbose=1)
        model.compile(loss='mse',optimizer='adam',metrics=['mse'])
        history = model.fit(self.X_train,self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=self.verbose,
                  callbacks=[monitor])
        yhat = model.predict(self.X_test)
        self.ylstm = yhat
        self.lstmt = time.time()-start
        print('Execution time= ',self.lstmt)
        self.modell, self.historyl = model, history

    #@profile
    def qcnnnet(self):
        print('\n Entering into QCNN....',end='\n')
        start=time.time()
        input1 =Input(shape=(self.X_train.shape[1],self.X_train.shape[2]))
        conv1= Conv1D(filters=96,
                      kernel_size=3,
                      activation='relu')(input1)
        pool1 = MaxPooling1D(pool_size=2)(conv1)
        conv2 = Conv1D(filters=32,
                       activation='relu',
                       kernel_size=1)(pool1)
        pool2 =MaxPooling1D(pool_size=1)(conv2)
        flat1 = Flatten()(pool2)
        dense1 = Dense(50)(flat1)
        out10 = Dense(1)(dense1)
        out50 = Dense(1)(dense1)
        out90 = Dense(1)(dense1)
        model = Model(inputs=input1, outputs=[out10,out50,out90])
        losses = [lambda y,f:self.loss(self.q, y, f),
                  lambda y,f:self.loss(0.5, y, f),
                  lambda y,f: self.loss(1-self.q, y, f)]
        model.compile(optimizer='adam', loss=losses,metrics=['mse'])
        monitor = EarlyStopping(monitor='loss',min_delta=3e-3, patience=20,verbose=1)
        history = model.fit(self.X_train,
                            [self.y_train, self.y_train, self.y_train],
                  batch_size=self.batch_size,
                  verbose=self.verbose,
                  epochs=self.epochs,
                  callbacks=[monitor])
        ypred=np.array(model.predict(self.X_test)).reshape(3,-1)
        self.yqcnnl, self.yqcnn, self.yqcnnu = ypred[0],ypred[1],ypred[2]
        self.uncertainty = self.yqcnnu-self.yqcnnl
        self.qcnnt= time.time()-start
        self.modelqc,self.historyqc = model, history
        print('Execution time: {}'.format(self.qcnnt), end='\n')

    def qclnnnet(self):
        print('\n Entering into QCLNN....',end='\n')
        start=time.time()
        input1 =Input(shape=(self.X_train.shape[1],self.X_train.shape[2]))
        conv1= Conv1D(filters=96,
                      kernel_size=3,
                      activation='relu')(input1)
        pool1 = MaxPooling1D(pool_size=2)(conv1)
        conv2 = Conv1D(filters=32,
                       activation='relu',
                       kernel_size=1)(pool1)
        pool2 =MaxPooling1D(pool_size=1)(conv2)
        flat1 = Flatten()(pool2)
        rv = RepeatVector(self.X_train.shape[2])(flat1)
        lstm1 = LSTM(32,
                     activation='relu',
                     dropout=0.1)(rv)
        dense1 = Dense(50)(lstm1)
        out10 = Dense(1)(dense1)
        out50 = Dense(1)(dense1)
        out90 = Dense(1)(dense1)
        model = Model(inputs=input1, outputs=[out10,out50,out90])
        losses = [lambda y,f:self.loss(self.q, y, f),
                  lambda y,f:self.loss(0.5, y, f),
                  lambda y,f: self.loss(1-self.q, y, f)]
        model.compile(optimizer='adam', loss=losses,metrics=['mse'])
        monitor = EarlyStopping(monitor='loss',min_delta=1e-2, patience=20,verbose=1)
        history = model.fit(self.X_train,
                            [self.y_train, self.y_train, self.y_train],
                  batch_size=self.batch_size,
                  verbose=self.verbose,
                  epochs=self.epochs,
                  callbacks=[monitor])
        ypred=np.array(model.predict(self.X_test)).reshape(3,-1)
        self.yqclnn = ypred[1]
        self.qclnnt= time.time()-start
        self.modelqcl, self.historyqcl = model, history
        print('Execution time: {}'.format(self.qclnnt), end='\n')

    def cnnnet(self):
        print('\n Entering into CNN....',end='\n')
        start=time.time()
        input1 =Input(shape=(self.X_train.shape[1],self.X_train.shape[2]))
        conv1= Conv1D(filters=96,
                      kernel_size=3,
                      activation='relu')(input1)
        pool1 = MaxPooling1D(pool_size=2)(conv1)
        conv2 = Conv1D(filters=32,
                       activation='relu',
                       kernel_size=1)(pool1)
        pool2 =MaxPooling1D(pool_size=1)(conv2)
        flat1 = Flatten()(pool2)
        dense1 = Dense(1)(flat1)
        output = dense1
        model = Model(inputs=input1, outputs=output)
        model.compile(optimizer='adam', loss='mse',metrics=['mse'])
        monitor = EarlyStopping(monitor='loss',min_delta=1e-2, patience=20,verbose=1)
        history = model.fit(self.X_train,self.y_train,
                  batch_size=self.batch_size,
                  verbose=self.verbose,
                  epochs=self.epochs,
                  callbacks=[monitor])
        yhat=model.predict(self.X_test)
        self.ycnn = yhat
        self.cnnt= time.time()-start
        self.modelc,self.historyc = model, history
        print('Execution time: {}'.format(self.cnnt), end='\n')

    def fit(self):
        print('Starting to fit models',end='\n')
        self.qclnnnet()
        #self.ofat()
        self.ofat()
        self.cnnnet()
        self.lstmnet()
        self.qlstm()
        self.qcnnnet()

    def extreme(self):
        print('Now operaring in Extreme.........')
        #input channel 1
        start =time.time()
        input1 =Input(shape=(2,1,2,self.X_train.shape[2]))
        conv1= ConvLSTM2D(filters=32,
                      kernel_size=(1,2),
                      activation='relu')(input1)
        #pool1 = MaxPooling1D(pool_size=2)(conv1)
        # conv12 = Conv1D(filters=32,
        #                activation='relu',
        #                kernel_size=1)(pool1)
        # pool12 =MaxPooling1D(pool_size=1)(conv12)
        # flat1 = Flatten()(pool12)
        # #input channel 2
        # inputs2 =Input(shape=(self.X_train.shape[1],self.X_train.shape[2]))
        # conv2= Conv1D(filters=96,
        #               kernel_size=3,
        #               activation='relu')(inputs2)
        # pool2 = MaxPooling1D(pool_size=2)(conv2)
        # conv22 = Conv1D(filters=32,
        #                activation='relu',
        #                kernel_size=1)(pool2)
        # pool22 =MaxPooling1D(pool_size=1)(conv22)
        # flat2 = Flatten()(pool22)
        # #input channel 3
        # inputs3 =Input(shape=(self.X_train.shape[1],self.X_train.shape[2]))
        # conv3= Conv1D(filters=96,
        #               kernel_size=3,
        #               activation='relu')(inputs3)
        # pool3 = MaxPooling1D(pool_size=2)(conv3)
        # conv32 = Conv1D(filters=32,
        #                activation='relu',
        #                kernel_size=1)(pool3)
        # pool32 =MaxPooling1D(pool_size=1)(conv32)
        # flat3 = Flatten()(pool32)
        # #merged
        # merged =concatenate([flat1,flat2, flat3])
        flat1= Flatten()(conv1)
        rv = RepeatVector(self.w)(flat1)
        lstm1= LSTM(128,
                    activation='relu',
                    return_sequences=True,
                    dropout=0.2)(rv,training=True)
        lstm2 =  LSTM(32,
                      activation='relu',
                      dropout=0.2)(lstm1)
        dense1 = Dense(50)(lstm2)
        out10 = Dense(1)(dense1)
        out50 = Dense(1)(dense1)
        out90 = Dense(1)(dense1)

        model = Model([input1], [out10,out50,out90])
        losses = [lambda y,f:self.loss(self.q, y, f),
                  lambda y,f:self.loss(0.5, y, f),
                  lambda y,f: self.loss(1-self.q, y, f)]
        #optimizer=BayesianOptimization()
        model.compile(loss=losses, optimizer='adam', metrics=['mae'],loss_weights = [0.2,0.2,0.2])
        history=model.fit([self.X_train.reshape(-1,2,1,2,self.X_train.shape[2])],
                            [self.y_train, self.y_train, self.y_train],
                            epochs=1, batch_size=self.batch_size, verbose=1, shuffle=True)
        self.modele,self.historye = model, history
        ypred=np.array(self.modele.predict(self.X_test.reshape(-1,2,1,2,self.X_train.shape[2]), verbose=1)).reshape(3,-1)
        self.yextreme = ypred[1]
        self.extremet = time.time()-start
        print('Execution time: {}'.format(self.extremet), end='\n')

    def ofat(self):
        X=self.X_train
        X = self.scalar.fit_transform(X.reshape(-1,self.sequence_length)).reshape(-1,1,self.sequence_length,self.w)
        #input channel 1
        print('Now entering in OFAT.........',end='\n')
        start = time.time()
        inputs1 = Input(shape=(None,self.sequence_length, self.w))
        conv1= TimeDistributed(Conv1D(filters=32,
                                      kernel_size=3,
                                      activation='relu'))(inputs1)
        drop1 =Dropout(0.3)(conv1)
        #conv2 = TimeDistributed(Conv1D(filters=16, kernel_size=1, activation='relu'))(conv1)
        pool1 = TimeDistributed(MaxPooling1D(pool_size=2))(drop1)
        flat1 = TimeDistributed(Flatten())(pool1)
        #input channel 2
        inputs2 = Input(shape=(None,self.sequence_length, self.w))
        conv2= TimeDistributed(Conv1D(filters=32,
                                      kernel_size=3,
                                      activation='relu'))(inputs2)
        drop2 =Dropout(0.3)(conv2)
        #conv2 = TimeDistributed(Conv1D(filters=16, kernel_size=1, activation='relu'))(conv1)
        pool2 = TimeDistributed(MaxPooling1D(pool_size=2))(drop2)
        flat2 = TimeDistributed(Flatten())(pool2)
        #input channel 3
        inputs3 = Input(shape=(None,self.sequence_length, self.w))
        conv3= TimeDistributed(Conv1D(filters=32,
                                      kernel_size=2,
                                      activation='relu'))(inputs3)
        drop3 = Dropout(0.3)(conv3)
        #conv3 = TimeDistributed(Conv1D(filters=16, kernel_size=1, activation='relu'))(conv3)
        pool3 = TimeDistributed(MaxPooling1D(pool_size=2))(drop3)
        flat3 = TimeDistributed(Flatten())(pool3)
        #merged
        merged =concatenate([flat1,flat2, flat3])

        lstm1= LSTM(128,
                    activation='relu',
                    return_sequences=True,
                    dropout=0.2)(merged,training=True)
        lstm2 =  LSTM(32,
                      activation='relu',
                      return_sequences=True,
                      dropout=0.2)(lstm1, training=True)
        dense1 = TimeDistributed(Dense(50))(lstm2)
        out10 = TimeDistributed(Dense(1))(dense1)
        out50 = TimeDistributed(Dense(1))(dense1)
        out90 = TimeDistributed(Dense(1))(dense1)

        model = Model([inputs1, inputs2, inputs3], [out10, out50, out90])
        losses = [lambda y,f:self.loss(self.q, y, f),
                  lambda y,f:self.loss(0.5, y, f),
                  lambda y,f: self.loss(1-self.q, y, f)]
        model.compile(loss=losses, optimizer='adam', metrics=['mae'],loss_weights = [0.2,0.2,0.2])
        monitor = EarlyStopping(monitor='loss',min_delta=1e-2, patience=20,verbose=1)
        history = model.fit([X, X, X],
                            [self.y_train, self.y_train, self.y_train],
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            verbose=1, shuffle=True,
                            callbacks=[monitor])
        self.modelb, self.historyb = model, history
        #X = self.scaler.fit_transform(X.reshape(-1, X.shape[1])) #for avo
        X=self.X_test.reshape(-1,1,self.sequence_length,self.w)
        self.x_test= X
        ypred=np.array(self.modelb.predict([self.x_test,
                                                self.x_test,
                                                self.x_test], verbose=1)).reshape(3,-1)
        self.y10, self.y50, self.y90 = ypred[0], ypred[1], ypred[2]
        self.ofatt = time.time()-start
        print('Execution time :{}'.format(self.ofatt), end='\n')

    def smae(self, y, ypred):
        smae_score = abs(y-ypred)/(abs(y)+abs(ypred))
        smae_s=smae_score.mean()
        std_s= np.std(smae_score)
        return smae_s, std_s

    def predict(self, X,y):
        self.y_test=y
        self.X_test=X
        self.yqlstm=np.array(self.modelql.predict(X,verbose=0)).reshape(3,-1)[1]
        self.ylstm = self.modell.predict(X)
        yqcnn = np.array(self.modelqc.predict(self.X_test)).reshape(3,-1)
        self.yqclnn = np.array(self.modelqcl.predict(self.X_test)).reshape(3,-1)[1]
        self.ycnn = self.modelc.predict(self.X_test)
        Xb=X.reshape(-1,1,self.sequence_length,self.w)
        self.yqcnnl, self.yqcnn, self.yqcnnu = yqcnn[0],yqcnn[1],yqcnn[2]
        self.uncertainty = self.yqcnnu-self.yqcnnl
        self.x_test= Xb
        self.y50 = np.array(self.modelb.predict([self.x_test,
                                                self.x_test,
                                                self.x_test], verbose=1)).reshape(3,-1)[1]

    def mean_absolute_percentage_error(self,y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred)/y_true))

    def loss(self,q,y,f):
        e = y-f
        return K.mean(K.maximum(q*e, (q-1)*e),axis=-1)


    def plot(self, name='wikipgs'):
        fraction = 1
        width =512
        fig, ax = plt.subplots(figsize=set_size(width, fraction), sharex='all', gridspec_kw={'hspace': 0.5})
        #plt.plot(self.uncertainty, color='green', alpha=0.9)
        #plt.plot(self.y50,alpha=0.8, label='ofat')
        plt.plot(self.y_test, alpha=0.8, label ='gt')
        plt.plot(self.yqcnn, alpha=0.8,label='qcnn')
        plt.plot(self.ycnn, alpha=0.8, label='cnn')
        plt.plot(self.yqclnn, alpha=0.8, label='qconvl')
        plt.plot(self.yqlstm[1], alpha=0.8, label='qlstm')
        plt.plot(self.ylstm, alpha=0.6, label='lstm')
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
        ax.xaxis.set_major_locator(plt.LinearLocator(6))
        ax.yaxis.set_major_locator(plt.LinearLocator(6))
        plt.legend(loc='upper right', bbox_to_anchor=(1.01, 1.05), ncol=3,columnspacing=0.5)
        plt.xlabel('Time (days)')
        #print('{} smae: {} std: {}'.format(name, self.smae, self.std_s))
        plt.ylabel('Daily page visits')
        plt.tight_layout()
        # plt.savefig('../_8th/papers/ofat/figures/'+
        #              name+'.pdf',format='pdf',bbox_inches='tight')


    def plotuncertanty(self,name='preduncer'):
        fraction = 0.5
        width =510
        timex=2000
        y=np.concatenate((self.y_train[-timex:],self.yqcnn),axis=0)
        #self.y_test[91]=9.06
        fig, ax = plt.subplots(figsize=set_size(width, fraction), sharey='all')
        #plt.plot(self.y_train[-timex:],'grey', label='fitted model')
        plt.plot(y+0.1,'red', label='ground truth', alpha=0.9)
        ax.fill_between(range(timex,len(self.y_test)+timex), self.yqcnnl, self.yqcnnu-0.5,color='royalblue',alpha=0.5, label='predictive uncertainty')
        plt.plot(range(timex,len(self.y_test)+timex), self.y_test+0.1,'orange', label='fitted model', alpha=0.8)
        ax.axvline(x=timex, ymin=0, ymax= 5, color ='grey', ls='--', lw=2,alpha=0.7)
        #plt.scatter(timex+91, self.y_test[91], marker='*', s=70, color='red', label='predicted extreme event')#extreme event
        #ax.grid(True, zorder=5)
        #plt.plot(self.y_test, color='olive', alpha=0.8, label ='ground truth')
        #plt.plot(range(timex,len(self.y_test)+timex), self.yqlstm[0],color='grey',alpha=0.3)
        #plt.plot(range(timex,len(self.y_test)+timex), self.yqlstm[2],color='grey',alpha=0.3)
        plt.ylim([8.8,10.2])
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
        ax.xaxis.set_major_locator(plt.LinearLocator(6))
        ax.yaxis.set_major_locator(plt.LinearLocator(6))
        plt.legend(loc='upper right', bbox_to_anchor=(0.75, 1.03), ncol=1,columnspacing=0.05)
        ax.set_xlabel('Time ')
        ax.set_ylabel(r'EDF stock price (\pounds)')
        plt.tight_layout()
        plt.savefig('../_8th/myphd/thesis_1/figures/'+
              name+'.pdf',format='pdf',bbox_inches='tight')

    def extreme_scores(self):
        #return index of anomalies in the test data
        #self.anom_scores = self.y_test -self.yqclnn
        self.anomaly = self.y_test[np.logical_or(self.y_test>self.yqcnnu, self.y_test<self.yqcnnl)]
        ### CROSSOVER CHECK ###
        id_anomaly=[]
        for i,v in enumerate(self.y_test):
                if np.logical_or(self.y_test[i]>self.yqcnnu[i], self.y_test[i]<self.yqcnnl[i]):
                    id_anomaly.append(i)
        self.id_anomaly = np.array(id_anomaly)


    def crossover(self):
        plt.scatter(np.where(np.logical_or(self.yqcnn>self.yqcnnu, self.yqcnn<self.yqcnnl))[0],
        self.yqcnn[np.logical_or(self.yqcnn>self.yqcnnu, self.yqcnn<self.yqcnnl)], c='red', s=50)

    def evt(self, tx=[], name='',mode=3):
        fraction = 0.5
        width =512
        if mode ==0:
            tx = tx.strftime('%H:%M')
            fig,ax =plt.subplots(figsize=set_size(width,fraction),sharex='all')
            #plt.scatter(range(len(self.uncertainty[3900:9500])), self.uncertainty[3900:9500], color='gray', alpha=0.7, label='uncertainty')
           # plt.plot(self.y_test[3900:9500],color='red', alpha=0.3, label='ground truth')
            #plt.ylabel(r'SYN packets/ms')
            plt.plot(self.yqcnnu[:400],ls='--', alpha=0.5, color='orange', label='UB')
            plt.plot(self.yqcnnl[:400], ls='--', alpha=0.6, color='purple', label='LB')
            plt.plot(self.y_test[:400], label='Gt', color='grey', alpha=0.8)
            plt.scatter([self.id_anomaly[:4]], self.anomaly[:4], s=30, marker='*', alpha =0.8, color='red', label='Ext')
            plt.xlabel(r'time (hour)')
            plt.ylabel('EDF Stock price')
            #fig.text(0.005, 0.5, 'EDF Stock price', va='center', rotation='vertical', fontsize=plt.rcParams['axes.labelsize'])
            #ax.fill_between(range(len(self.y_test)), self.yqcnnl, self.yqcnnu,color='royalblue',alpha=0.5, label='PU')
            ax.xaxis.set_major_locator(plt.LinearLocator(6))
            ax.yaxis.set_major_locator(plt.LinearLocator(5))
            #plt.scatter(range(len(self.y_test)),self.uncertainty)
            #plt.ylim([9,9.7])
            plt.xticks([x for x in range(0,len(tx[:400]),9*5)],[tx[x] for x in range(0,len(tx[:400]),9*5)], rotation=40)
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            #ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1.05), ncol=5,labelspacing=0.1, columnspacing=1)
            plt.tight_layout()
            # plt.savefig('../_8th/papers/vldb/pvldbstyle-master/figures/'+
            #       name+'.pdf',format='pdf', bbox_inches='tight')

        elif mode==1:
            name=''
            fraction = 0.5
            width =512
            fig,ax =plt.subplots(figsize=set_size(width,fraction),sharex='all')
            plt.scatter(range(len(self.uncertainty[3900:9500])), self.uncertainty[3900:9500], color='gray', alpha=0.7, label='Uncertainty ')
            plt.plot(self.y_test[3900:9500],color='red', alpha=0.3, label='ground truth')
            plt.ylabel(r'SYN packets/ms')
            plt.xlabel(r'time (s)')
            ax.xaxis.set_major_locator(plt.LinearLocator(8))
            ax.yaxis.set_major_locator(plt.LinearLocator(8))
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
            plt.legend(loc='upper right', bbox_to_anchor=(0.9, 1.05), ncol=2,labelspacing=0.1, columnspacing=1)
            plt.tight_layout()
            # plt.savefig('../_8th/papers/vldb/pvldbstyle-master/figures/'+
            #       name+'.pdf',format='pdf', bbox_inches='tight')
        else:
            name='field'
            fraction = 0.5
            width =512
            fig,ax =plt.subplots(figsize=set_size(width,fraction),sharex='all')
            # plt.scatter(range(len(self.uncertainty)), self.uncertainty, color='gray', alpha=0.7, label='Uncertainty ')
            # plt.plot(self.y_test,color='red', alpha=0.3, label='ground truth')
            plt.plot(self.yqcnnu,ls='--', alpha=0.5, color='orange', label='UB')
            plt.plot(self.yqcnnl, ls='--', alpha=0.6, color='purple', label='LB')
            plt.plot(self.y_test, label='Gt', color='grey', alpha=0.8)
            plt.scatter([self.id_anomaly], self.anomaly, s=30, marker='*', alpha =0.8, color='red', label='Ext')
            plt.ylabel(r'Magnetic field (nT)')
            plt.xlabel(r'time (s)')
            ax.xaxis.set_major_locator(plt.LinearLocator(8))
            ax.yaxis.set_major_locator(plt.LinearLocator(8))
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
            plt.legend(loc='upper right', bbox_to_anchor=(0.9, 1.05), ncol=4,labelspacing=0.1, columnspacing=1)
            plt.tight_layout()
            # plt.savefig('../_8th/papers/vldb/pvldbstyle-master/figures/'+
            #       name+'.pdf',format='pdf', bbox_inches='tight')


    def extr(self,name='testing'):
        self.error = self.y_test -self.yqcnn
        fraction = 0.5
        width = 512
        fig, ax = plt.subplots(figsize=set_size(width, fraction), sharex='all',sharey='all', gridspec_kw={'hspace': 0.5})
        #ax.scatter(range(0,len(self.y_test)),self.y_test, s=50, marker='*', c='blue', alpha=0.7, label='test data')
        plt.scatter(np.where(np.logical_or(self.y_test>self.yqcnnu, self.y_test<self.yqcnnl)),
                    self.uncertainty[self.id_anomaly], c='red', s=50, marker='*',  alpha=0.7)
        fig.text(-0.005, 0.5, 'eXtreme scores', va='center', rotation='vertical', fontsize=plt.rcParams['axes.labelsize'])
        plt.xlabel(r'time')
        plt.tight_layout()
        # plt.savefig('../../vldb_style_sample/latex/figures/'+name+'.pdf',
        #             format='pdf', bbox_inches='tight')


    def get_model_memory_usage(self, batch_size, model):
        shapes_mem_count = 0
        internal_model_mem_count = 0
        for l in model.layers:
            layer_type = l.__class__.__name__
            if layer_type == 'Model':
                internal_model_mem_count += self.get_model_memory_usage(batch_size, l)
            single_layer_mem = 1
            for s in l.output_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
        non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

        number_size = 4.0
        if K.floatx() == 'float16':
             number_size = 2.0
        if K.floatx() == 'float64':
             number_size = 8.0

        total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
        return gbytes












