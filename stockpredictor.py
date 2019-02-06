# Import
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from numpy import newaxis
import matplotlib.pyplot as plt
import quandl
import datetime as dt

quandl.ApiConfig.api_key='sn3psDvBzGao3UpM6QE5'

#gets closing price of a given stock
def indiv_stock(symbol,start,end):
	mydata = quandl.get('EOD/'+str(symbol), start_date=str(start), end_date=str(end),column_index='4')
	return mydata

#get 5 years of historical data
end_date=dt.date.today()-dt.timedelta(days=1)
before=dt.timedelta(days=1825)
start_date=end_date-before
stock_price=indiv_stock("MSFT",start_date,end_date)
print(stock_price)
plt.plot(stock_price)
plt.show()

# Scale data
data=np.asarray(stock_price)
data=data.reshape(data.shape[0],1)
scaler = MinMaxScaler(feature_range=(0, 1))
data=scaler.fit_transform(data)

# Training and test data
# Create a function to process the data into 7 day look back slices
def processData(data,lb):
    X,Y = [],[]
    for i in range(len(data)-lb):
        X.append(data[i:(i+lb),0])
        Y.append(data[(i+lb),0])
    return np.array(X),np.array(Y)
X,y = processData(data,1)
X_train,X_test = X[:int(np.floor(X.shape[0]*0.80))],X[int(np.floor(X.shape[0]*0.80)):]
y_train,y_test = y[:int(np.floor(X.shape[0]*0.80))],y[int(np.floor(X.shape[0]*0.80)):]


#Build the model
model = Sequential()
model.add(LSTM(256,input_shape=(1,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
#Reshape data for (Sample,Timestep,Features) 
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
#Fit model with history to check for overfitting
history = model.fit(X_train,y_train,epochs=300,validation_data=(X_test,y_test),shuffle=False)

act = []
pred = []
#for i in range(250):
Xt = model.predict(X_test[-1].reshape(1,1,1))
last_val_scale=Xt/Xt
next_val=model.predict(np.reshape(Xt, (1,1,1)))
print("Next day value:" + str(np.asscalar(scaler.inverse_transform(next_val))))
print('predicted:{0}, actual:{1}'.format(scaler.inverse_transform(Xt),scaler.inverse_transform(y_test[-1].reshape(-1,1))))
pred.append(scaler.inverse_transform(Xt))
act.append(scaler.inverse_transform(y_test[-1].reshape(-1,1)))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
Xt = model.predict(X_test)
plt.plot(scaler.inverse_transform(y_test.reshape(-1,1)))
plt.plot(scaler.inverse_transform(Xt))
plt.show()
