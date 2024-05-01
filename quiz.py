import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load your time series dataset
# Assuming you have a pandas DataFrame with one column representing the time series data
# Replace 'your_dataset.csv' with the path to your dataset
data = pd.read_csv('your_dataset.csv')

# Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to create dataset for time series forecasting
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Choose time step
time_step = 100
X, y = create_dataset(scaled_data, time_step)

# Reshape input data to be 3-dimensional in the form [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Splitting dataset into train and test sets
train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(data)]
y_train, y_test = y[0:train_size], y[train_size:len(data)]

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])

# Evaluate the model
train_score = np.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
print('Train Score: %.2f RMSE' % (train_score))
test_score = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
print('Test Score: %.2f RMSE' % (test_score))

# Visualize results
plt.figure(figsize=(10,6))
plt.plot(data.index[:-time_step-1], scaler.inverse_transform(data.values[:-time_step-1]), label='Original data')
plt.plot(data.index[time_step:train_size], train_predict, label='Train predictions')
plt.plot(data.index[train_size+time_step+1:], test_predict, label='Test predictions')
plt.legend()
plt.show()
