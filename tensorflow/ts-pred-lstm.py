import tensorflow as tf
import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.ion()


dataset = pd.read_csv( "~/Downloads/international-airline-passengers.csv" 
        , usecols=[1]
        , engine='python'
        , skipfooter=3
        )

plt.plot( dataset )

np.random.seed(123)

dataset = dataset.values.astype('float32')

dataset = MinMaxScaler( feature_range=(0,1) ).fit_transform( dataset )

plt.plot( dataset )
plt.clf()

train_size = int( len(dataset)*0.67 )
test_size = len(dataset) - train_size
train = dataset[0:train_size,:]
test = dataset[train_size:len(dataset),:]

plt.plot( train )
plt.plot( test )

def create_dataset( dataset , look_back=1 ):
    dataX = []
    dataY = []
    for i in range( len(dataset) - look_back - 1):
        dataX.append( dataset[i:(i+look_back) , 0] )
        dataY.append( dataset[i + look_back , 0] )
    return np.array( dataX ) , np.array( dataY )

look_back = 1
trainX , trainY = create_dataset( train , look_back )
testX , testY = create_dataset( test , look_back )

plt.clf()
plt.plot( testX )
plt.plot( testY )

# [ samples , time steps , features ]
trainX = np.reshape( trainX , (trainX.shape[0] , 1 , trainX.shape[1] ) )
testX = np.reshape( testX , (testX.shape[0] , 1 , testX.shape[1] ) )

model = Sequential()
model.add( LSTM( 4 , input_dim=look_back ) )
model.add( Dense(1) )
model.compile( loss='mean_squared_error' , optimizer='adam' )
model.fit( trainX , trainY , nb_epoch=100 , batch_size=1 , verbose=2 )

trainPredict = model.predict( trainX )
testPredcit = model.predict( testX )

plt.clf()
plt.plot( trainPredict )
plt.plot( testPredcit )

model.summary()

model.get_config()

model.get_weights()

model.to_json()
model.to_yaml()

model_lstm = Sequential()
model_lstm.add( LSTM( 4, input_dim=2, init='zero', inner_init='zero') )
model_lstm.summary()
model_lstm.get_config()

for w in model_lstm.get_weights():
    print( np.shape( w ) )
