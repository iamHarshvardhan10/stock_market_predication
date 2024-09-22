from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy
import pandas as pd
import math
from sklearn.linear_model import ElasticNet
import os

# setting a seed for reproducibility
numpy.random.seed(10)
# read all stock files in directory indivisual_stocks_5yr


def read_all_stock_files(folder_path):
    allFiles = []
    for (_, _, files) in os.walk(folder_path):
        allFiles.extend(files)
        break

    dataframe_dict = {}
    for stock_file in allFiles:
        df = pd.read_csv(folder_path + "/" + stock_file)
        dataframe_dict[(stock_file.split('_'))[0]] = df

    return dataframe_dict
# convert an array of values into a dataset matrix


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
# create dataset from the dataframe


def create_preprocessed_Dataset(df):
    # Ensure the DataFrame contains the necessary columns
    if not all(col in df.columns for col in ['date', 'open']):
        raise ValueError("DataFrame must contain 'date' and 'open' columns")

    # Select relevant columns and convert 'open' to numeric
    df = df[['date', 'open']]
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df.dropna(subset=['open'], inplace=True)  # This should be fine

    dataset = df['open'].values
    dataset = dataset.reshape(-1, 1).astype('float32')

    # Split into train and test sets
    train_size = len(dataset) - 2
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # Reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    return trainX, trainY, testX, testY


def getData(df):
    # Create the lists / X and Y data sets
    dates = []
    prices = []

    # Get the number of rows and columns in the data set
    # df.shape

    # Get the last row of data (this will be the data that we test on)
    last_row = df.tail(1)

    # Get all of the data except for the last row
    df = df.head(len(df) - 1)
    # df

    # The new shape of the data
    # df.shape

    # Get all of the rows from the Date Column
    df_dates = df.loc[:, 'date']
    # Get all of the rows from the Open Column
    df_open = df.loc[:, 'open']

    # Create the independent data set X
    for date in df_dates:
        dates.append([int(date.split('-')[2])])

    # Create the dependent data se 'y'
    for open_price in df_open:
        prices.append(float(open_price))

    # See what days were recorded
    last_date = int(((list(last_row['date']))[0]).split('-')[2])
    last_price = float((list(last_row['open']))[0])
    return dates, prices, last_date, last_price


def LSTM_model(dates, prices, test_date, df):
    df.drop(df.columns.difference(['date', 'open']), axis=1, inplace=True)
    df = df['open']
    dataset = df.values
    dataset = dataset.reshape(-1, 1)
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = len(dataset) - 2
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train, X_test, y_train, y_test = train_test_split(
        trainX, trainY, test_size=0.33, random_state=42)
    # reshape input to be [samples, time steps, features]
    X_train = numpy.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = numpy.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

    # make predictions
    trainPredict = model.predict(X_train)
    mainTestPredict = model.predict(X_test)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    y_train = scaler.inverse_transform([y_train])

    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    mainTestPredict = scaler.inverse_transform(mainTestPredict)
    mainTestPredict = [item for sublist in mainTestPredict for item in sublist]
    y_test = scaler.inverse_transform([y_test])
    test_score = mean_squared_error(y_test[0], mainTestPredict)
    # calculate root mean squared error
    trainPredict = [item for sublist in trainPredict for item in sublist]

    # print(trainPredict, testPredict[0])

    return (trainPredict, (testPredict[0])[0], test_score)
