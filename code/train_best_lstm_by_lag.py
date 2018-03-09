import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Run LSTM.')
parser.add_argument('path', help='File path')
parser.add_argument('result_path', help='Results file path')
parser.add_argument('--epochs', default=1, help="Input Values", type=int)
parser.add_argument('--stateful', default=False, help="Stateful Keras LSTM", type=str2bool)
args = parser.parse_args()

#Imports
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
import sklearn.model_selection as ms
import sklearn.metrics as metrics

import sys
sys.path.append('../lib')
from helpers import *
import os
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import backend as K


"""
Function to create lstm model
"""
def create_lstm(input_dim,output_dim,timesteps, nodes,loss='mean_squared_error',optimizer='adam',activation="tanh",recurrent_activation='hard_sigmoid', batch_size = 168,stateful=False):
    model = Sequential()
    model.add(LSTM(nodes, input_shape=(timesteps,input_dim),activation=activation, recurrent_activation=recurrent_activation,stateful=stateful, batch_size=batch_size))
    model.add(Dense(output_dim))
    model.compile(loss=loss, optimizer=optimizer)
    return model


#Reading results file
results_df = pd.read_csv(args.result_path,index_col=0)
#Reading data files
data =   pd.Series.from_csv(args.path)

print("===== Stateful: "+str(args.stateful))

best_scores = []
for i,result_args in results_df.iterrows():
    input_dim = result_args.input_dim
    output_dim = result_args.output_dim
    
    timesteps = result_args.timesteps
    X,y = create_data_cube(data, input_dim=input_dim,timesteps=timesteps)

    arg = result_args
    print(arg)
    preprocess = arg.preproc
    source = arg.source

    train_perc = 0.8
    trainlen = int(train_perc*len(X))
    trainlen = trainlen - (trainlen%arg.batch_size)
    testlen =  len(X) - trainlen
    testlen = testlen - (testlen%arg.batch_size)

    #Dividing Train and Test
    X_train,X_test = X[:trainlen], X[trainlen:trainlen+testlen]
    y_train,y_test = y[:trainlen], y[trainlen:trainlen+testlen]
    y_train_orig  = y_train


    if "minmax" in preprocess:
        print("Using Minmax Scaler with feature range ", end="")
        if "1" in preprocess:
            print(" (0,1) ")
            feature_range=(0,1)
        else:
            print( " (-1,1) " )
            feature_range=(-1,1)
        minmax_in = preprocessing.MinMaxScaler(feature_range=feature_range)
        minmax_out = preprocessing.MinMaxScaler(feature_range=feature_range)

        minmax_in.fit(X_train[:,0,:])
        minmax_out.fit(y_train)

        preproc_in = minmax_in
        preproc_out = minmax_out
 
    else:
        print("Using Standarization")
        standarization_in = preprocessing.StandardScaler()
        standarization_out = preprocessing.StandardScaler()

        standarization_in.fit(X_train[:,0,:])
        standarization_out.fit(y_train)

        preproc_in = standarization_in
        preproc_out = standarization_out

    for i in range(timesteps):
       X_train[:,i,:] = preproc_in.transform(X_train[:,i,:]) if preproc_in else X_train
       X_test[:,i,:] = preproc_in.transform(X_test[:,i,:]) if preproc_in else X_test
    y_train = preproc_out.transform(y_train) if preproc_out else y_train
    #Creating Model
    arg = arg.drop(["preproc","validation_score","train_score","source","stateful"])
    model = create_lstm(stateful = args.stateful,**arg)
    t0 = time.time()
    model.fit(X_train,y_train,epochs=args.epochs,batch_size=arg.batch_size, verbose=True, shuffle=False)
    t1 = time.time() - t0

    #Predicting Train
    y_approx = model.predict(X_train, batch_size=arg.batch_size)
    y_approx = preproc_out.inverse_transform(y_approx)
    score_train = metrics.mean_squared_error(y_train_orig,y_approx)

    #Predicting Test
    y_test_approx = model.predict(X_test, batch_size= arg.batch_size)
    y_test_approx = preproc_out.inverse_transform(y_test_approx)
    score_test = metrics.mean_squared_error(y_test,y_test_approx)

    print("Score train:", score_train, " Score test:", score_test )

    base = "../results/best_lstm/"
    base = base + "stateful/" if args.stateful else base
    os.makedirs(base,exist_ok=True)
    best_scores.append({"score_test":score_test, "score_train": score_train, "lags": result_args.input_dim, "time": t1})
    model.save(base + source + "_{}_lags_{}_timesteps_model.h5".format(result_args.input_dim,result_args.timesteps))
    np.savetxt(base + "lstm_y_train_{}_{}_lags_{}_timesteps.csv".format(source,input_dim, result_args.timesteps),y_train_orig)
    np.savetxt(base + "lstm_y_test_{}_{}_lags_{}_timesteps.csv".format(source,input_dim, result_args.timesteps),y_test)


    np.savetxt(base + "lstm_y_approx_train_{}_{}_lags_{}_timesteps.csv".format(source,result_args.input_dim, result_args.timesteps),y_approx)
    np.savetxt(base + "lstm_y_approx_test_{}_{}_lags_{}_timesteps.csv".format(source,result_args.input_dim, result_args.timesteps),y_test_approx)

    del model
    K.clear_session()

best_scores = pd.DataFrame(best_scores)
best_scores.to_csv(base + "lstm_{}_best_scores.csv".format(source))

