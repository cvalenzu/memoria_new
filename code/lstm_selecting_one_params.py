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
parser.add_argument('--inputs', default=1, help="Input vector size", type=int)
parser.add_argument('--outputs', default=12, help="Output vector size", type=int)
parser.add_argument('--timesteps', default=1, help="Timesteps", type=int)
parser.add_argument('--train_perc', default=0.8, help="Percentage of the data used to train the model. Value between 0 and 1. Default:0.8", type=float)
parser.add_argument('--batch_size', default=168, help="Batch size used to train LSTM",type=int)
parser.add_argument('--epochs', default=10, help="Input Values", type=int)
parser.add_argument('--preprocess', default="minmax_1", help="minmax or standarization")
parser.add_argument('--stateful', default=False, help="Stateful Keras LSTM", type=str2bool)
parser.add_argument('--verbose', default=False, help="Verbose Keras LSTM", type=str2bool)
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

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import backend as K

dataPath = args.path#"../data/canela.csv"
input_dim = args.inputs
output_dim = args.outputs
timesteps = args.timesteps
train_perc = args.train_perc
batch_size = args.batch_size
epochs = args.epochs
preprocess = args.preprocess
stateful = args.stateful
verbose = args.verbose

filename = os.path.basename(dataPath).replace(".csv","")
file_path = "../results/params_lstm_recursive/{}_scores_lstm_{}_lags_{}_outs_{}_timesteps_{}_batch_{}_preprocess_{}_stateful.csv".format(filename,input_dim, output_dim,timesteps,batch_size, preprocess, stateful)

if os.path.exists(file_path):
	print("File Exists, Skipping")
	exit(0)

def create_lstm(input_dim,output_dim,timesteps, nodes,loss='mean_squared_error',optimizer='adam',activation="tanh",recurrent_activation='hard_sigmoid', batch_size = 168,stateful=False):
    model = Sequential()
    model.add(LSTM(nodes, input_shape=(None,input_dim),activation=activation, recurrent_activation=recurrent_activation,stateful=stateful, batch_size=batch_size))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimizer)
    return model


def n_predict(model,X,steps=12, batch_size=168):
    y = np.empty((len(X),steps))
    X_tmp = np.empty((X.shape[0], X.shape[1]+steps,X.shape[2]))
    X_tmp[:,:X.shape[1], :] = X
    for i in range(steps):
        X_iter = X_tmp[:,:X.shape[1]+i,:]
        y[:,i] = model.predict(X_iter,batch_size = batch_size)[:,0]
        X_tmp[:,X.shape[1]+i,0] = y[:,i]
    return y


print("Reading Data")
#Data split parameters
data = pd.Series.from_csv(dataPath)

print("Preparing data")
X,y = create_data_cube(data, input_dim=input_dim,timesteps=timesteps)
y_multi = np.copy(y)
y = y[:,0].reshape((-1,1))

trainlen1 = int(train_perc*len(X))
trainlen = int(train_perc*trainlen1)
print("Removed from train ", (trainlen%batch_size), " values")
trainlen = trainlen - (trainlen%batch_size)
vallen = trainlen1 - trainlen
print("Removed from validation ", (vallen%batch_size), " values")
vallen = vallen - (vallen%batch_size)

X_train,X_test = X[:trainlen], X[trainlen:trainlen+vallen]
y_train,y_test = y[:trainlen], y[trainlen:trainlen+vallen]
y_train_orig  = y_train
y_train_multi, y_test_multi = y_multi[:trainlen], y_multi[trainlen:trainlen+vallen]

print("Preprocessing Data")

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


print("Creating Param List")
lsm_nodes = [32]
loss = ["mean_squared_error"]
activation = ["relu", "tanh", "sigmoid"]
recurrent_activation = ["hard_sigmoid","sigmoid"]
param_grid = {"nodes":lsm_nodes,"loss":loss, "input_dim":[input_dim], "output_dim": [12], "timesteps":[timesteps],
              "activation":activation, "recurrent_activation":recurrent_activation, "batch_size":[batch_size]}
params = ms.ParameterGrid(param_grid)


print("Evaluating Models")

scores = []
for param in params:
    print(param)
    np.random.seed(42)
    model = create_lstm(**param)
    model.fit(X_train, y_train,shuffle=False,verbose=verbose, epochs=epochs, batch_size=batch_size)
    try:
        y_approx_train = n_predict(model,X_train,batch_size=batch_size)
        for i in range(output_dim):
            y_approx_train[:,i] = preproc_out.inverse_transform(y_approx_train[:,i].reshape((-1,1))).reshape(-1)

        score = metrics.mean_squared_error(y_train_multi, y_approx_train)

        print("Score Train: ",score, end=" ")
        param["train_score"] = score

        y_approx = n_predict(model,X_test,batch_size=batch_size)
        for i in range(output_dim):
            y_approx[:,i] =  preproc_out.inverse_transform(y_approx[:,i].reshape((-1,1))).reshape(-1)
        score = metrics.mean_squared_error(y_test_multi,y_approx)

        print("Score Validation: ",score)
        param["validation_score"] = score
        scores.append(param)
  
    except:
        print("Error")    
    del model
    K.clear_session()
        

scores = pd.DataFrame(scores)


scores["source"] = filename
scores["stateful"] = stateful
scores["preproc"] = preprocess
os.makedirs("../results/params_lstm_recursive",exist_ok=True)
scores.to_csv("../results/params_lstm_recursive/{}_scores_lstm_{}_lags_{}_outs_{}_timesteps_{}_batch_{}_preprocess_{}_stateful.csv".format(filename,input_dim, output_dim,timesteps,batch_size, preprocess, stateful))
