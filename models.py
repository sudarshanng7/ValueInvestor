import random
import datetime
import numpy as np
import pandas as pd
import pmdarima as pm
import tensorflow as tf
#from fbprophet import Prophet
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
#from keras.callbacks import EarlyStopping, ModelCheckpoint

# Set a seed value
seed_value= 12321 


def train_test_split(df):
    """
    This function will take df and split the data to train test datasets based on the date.
    """
    # because this function depends on the analysis report function output, we need to take the difference instead of the price column, sins the difference column is the stationary price column version.
    df['Price'] = df['difference']
    df.drop('Date', axis=1, inplace=True)
    df = df.reset_index()
    # make the predictions only on the 2021 data point
    test = df[df.Date.astype(str).str.contains('2021')]
    test.drop(['Date', 'year', 'difference'], axis=1, inplace=True)
    test = test.reset_index(drop=True)
    # all the data points before 2021 will be in the training dataset
    train = df[df.Date.astype(str).str.contains("2021") == False]
    train.drop(['Date', 'year', 'difference'], axis=1, inplace=True)
    train = train.reset_index(drop=True)
    # train/ test split   
    X_train = train.drop('Price', axis=1)
    y_train = train['Price']
    X_test = test.drop('Price', axis=1)
    y_test = test['Price']
    return X_train, y_train, X_test, y_test

def actual_prediction_plot(actual, prediction, data_name, model_name):
    """
    param actual: is the real test price data.
    param prediction: the model prediction by using the actual test data.
    param data_name: used to naming the title figures.
    param model_name: to add the model with it's score to a dict
    This function will return Actual  VS. Predicted Price plot.
    """
    if isinstance(prediction, pd.Series):
        prediction = prediction.reset_index(drop=True)
    plt.figure(figsize=(16, 5))
    plt.plot(actual, label='Actual', linewidth=1.5)
    plt.plot(prediction, label='Predictions', linewidth=1.5)
    plt.title(f'Actual  VS. Predicted Price for {model_name}', fontsize=20)
    plt.legend();

def find_mae(y_true, y_pred):
    """
    return the mean absolute error
    """
    return mean_absolute_error(y_true, y_pred)

def arima_model(data_name, y_train, y_test):
    """
    param data_name: used to naming the title figures.
    param train: train dataset.
    param test: test dataset.
    This function will return the actual vs. predicted price, as well as the model name and the model mean 
    square error score.
    """
    
    # 1) create the model
    arima_model = pm.auto_arima(y_train, start_p=1, start_q=1, test='adf', max_p=6, max_q=6, 
                      m=1, d=None, seasonal=False, start_P=0, D=1, trace=False, error_action='ignore',  
                      suppress_warnings=True, stepwise=False)

    # 2) find the predicted price
    y_pred = arima_model.predict(len(y_test)) 
    model_name = f"{data_name} ARIMA Model"
    
    # 3) find the mae score
    mae = find_mae(y_test, y_pred)
    
    # 4) plot the actual vs. predicted price
    actual_prediction_plot(y_test, y_pred, data_name, model_name)
    return model_name, mae, y_pred

def sarima_model(data_name, y_train, y_test):
    """
    param data_name: used to naming the title figures.
    param train: train dataset.
    param test: test dataset.
    This function will return the actual vs. predicted price, as well as the model name and the model mean 
    square error score.
    """
    # 1) create the model
    sarima_model = pm.auto_arima(y_train, start_p=1, start_q=1, test='adf', max_p=6, max_q=6, 
                  m=1, d=None, seasonal=True, start_P=0, D=1, trace=False, error_action='ignore',  
                  suppress_warnings=True, stepwise=False)
     
    # 2) find the predicted price
    y_pred = sarima_model.predict(len(y_test))
    model_name = f"{data_name} SARIMA Model"
    
    # 3) find the mae score
    mae = find_mae(y_test, y_pred)
    
    # 4) plot the actual vs. predicted price
    actual_prediction_plot(y_test, y_pred, data_name, model_name)
    return model_name, mae, y_pred


def arimax_model(data_name, X_train, y_train, X_test, y_test):
    """
    param data_name: used to naming the title figures.
    param train: train dataset.
    param test: test dataset.
    This function will return the actual vs. predicted price, as well as the model name and the model mean 
    square error score.
    """

    # 1) create the model
    arimax_model = pm.auto_arima(y_train, exogenous=X_train, trace=False, error_action="ignore", 
                                 suppress_warnings=True)
    arimax_model.fit(y_train, exogenous=X_train)
    
    # 2) find the predicted price
    y_pred = arimax_model.predict(n_periods=len(y_test), exogenous=X_test)
    model_name = f"{data_name} ARIMAX Model"
    
    # 3) find the mae score
    mae = find_mae(y_test, y_pred)
    
    # 4) plot the actual vs. predicted price
    actual_prediction_plot(y_test, y_pred, data_name, model_name)
    return model_name, mae, y_pred

'''
def prophet_model(df, data_name):
    """
    param data_name: used to naming the title figures.
    This function will return the actual vs. predicted price, as well as the model name and the model mean 
    square error score.
    """    
    # here we need to go back to the original dataset
    df = df.reset_index()
    # 1) split the data to train, test datasets
    test = df[df.Date.astype(str).str.contains('2021')]
    train = df[df.Date.astype(str).str.contains("2021") == False]
    
    # 2) trake only the Date and Price columns for train dataset
    train = train[['Date', 'difference']]
    train.reset_index(drop=True, inplace=True)
    train.rename(columns={"Date": "ds", "difference": "y"}, inplace=True)
    
    # 3) trake only the Date and Price columns for test dataset
    test = test[['Date', 'difference']]
    test.reset_index(drop=True, inplace=True)
    test.rename(columns={"Date": "ds", "difference": "y"}, inplace=True)
    
    # 4) define and fit the model
    model = Prophet().fit(train)

    # 5) use the model to make a forecast
    forecast = model.predict(test[['ds']])
    model_name = f"{data_name} Prophet Model"
    # 6) find the mae score
    mae = find_mae(test.y, forecast.yhat)
    
    # 7) plot the actual vs. predicted price
    actual_prediction_plot(test.y, forecast.yhat, data_name, model_name)
    return model_name, mae, forecast.yhat
'''


def univariate_Lookback_data(df, lookback):
    """
    param lookback: how many days do you want as features.
    This function will return a new df with rows equal to df rows and with columns equal to the lookback num.
    """
    # transform the df to array
    data_raw = df.to_numpy()
    data = []
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback + 1])
    data = np.array(data)
    # define the features
    X = data [:, :-1].reshape(len(data), lookback)
    # define the target
    y = data[:, -1].reshape(-1,1)
    # transfor the array back to df
    df_featured = pd.DataFrame(X)
    df_featured['y'] = y
    df_featured.set_index(df.index[lookback:], inplace = True)
    # features/target split
    features = df_featured.drop(['y'], axis = 1)
    target = df_featured[['y']]
    return features, target

def multivariate_Lookback_data(features, target, n_steps):
    """
    param features: the X or feature columns
    param target: the y or the target column
    param n_step: lookback days num>
    """
    X, y = list(), list()
    for i in range(len(features)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(features):
            break
        # gather input and output parts of the pattern
        seq_x = features[i:end_ix, :]
        seq_y = target[end_ix-1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def train_valid_loss(hist, data_name, model_name):
    """
    param hist: output from deep NN like LSTM, and it contains both train and val loss scores.
    param data_name: used to naming the title figures.
    This function will return Train VS. Valid Loss plot.
    """    
    plt.figure(figsize=(16, 5))
    plt.plot(hist.history['loss'], label='Train loss', linewidth=1.5)
    plt.plot(hist.history['val_loss'], label='Valid loss', linewidth=1.5)
    plt.title(f'Train VS. Valid Loss for {model_name}', fontsize=20)
    plt.legend();

    
def LSTM(X_train, y_train, X_val, y_val, X_test, y_test, n_features, data_name, univariate=True):
    """
    This function will use the LSTM and return both of the model name as well as the mean square error score to 
    add it later to dict.
    """
    # adjust the seeds
    tf.random.set_seed(seed_value)
    
    # because we will use the LSTM with one and later with multiple features we need to define the model name to save it.    
    if univariate:
            model_name = f"{data_name} Univariate LSTM Model"
    else:
        model_name = f"{data_name} Multivariate LSTM Model"

    # 1) Create the model
    model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=100, activation='relu', input_shape=(X_train.shape[1], n_features)),
    tf.keras.layers.Dense(units=1)])

    # 2) Compile the model
    model.compile(loss=tf.keras.metrics.mae, optimizer=tf.keras.optimizers.Adam())

    # 3) Fit the model
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',patience=50),
        tf.keras.callbacks.ModelCheckpoint(f'output/models/{model_name}.h5', verbose=0, save_best_only=True)]
    hist = model.fit(X_train, y_train, epochs =5000, batch_size=16, validation_data=(X_val, y_val), callbacks=[callbacks], verbose=0)
    
    # 4) plot the train and val loss scores
    train_valid_loss(hist, data_name, model_name)
    # 5) find the predictions
    predictions = model.predict(X_test)
    # 6) find the mean square error score
    mae = find_mae(y_test, predictions)
    # 7) plot the actual vs. predicted price
    actual_prediction_plot(y_test, predictions, data_name, model_name)

    
    return model_name, mae, predictions

def ConvLSTM(X_train, y_train, X_val, y_val, X_test, y_test, data_name, input_shape, multivariate=True):
    """
    This function will use the ConvLSTM and return both of the model name as well as the mean square error score to 
    add it later to dict.
    """    
    tf.random.set_seed(seed_value)
    # because we will use the ConvLSTM with one and later with multiple features we need to define the model name to save it.    
    if multivariate:
        model_name = f"{data_name} Multivariate ConvLSTM Model"
    else:
        model_name = f"{data_name} Univariate ConvLSTM Model"    
    # 1) Create the model
    model = tf.keras.Sequential([
    tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=input_shape),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1)])

    # 2) Compile the model
    model.compile(loss=tf.keras.metrics.mae, optimizer=tf.keras.optimizers.Adam())

    # 3) Fit the model
    callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',patience=50),
    tf.keras.callbacks.ModelCheckpoint(f'output/models/{model_name}.h5', verbose=0, save_best_only=True)]
    hist = model.fit(X_train, y_train, epochs =5000, batch_size=16, validation_data=(X_val, y_val), callbacks=[callbacks], verbose=0)

    # 4) plot the train and val loss scores

    train_valid_loss(hist, data_name, model_name)

    # 5) find the predictions
    predictions = model.predict(X_test)
    # 6) find the mean square error score
    mae = find_mae(y_test, predictions)
    
    # 7) plot the actual vs. predicted price
    actual_prediction_plot(y_test, predictions, data_name, model_name)

    
    return model_name, mae, predictions

def scores(scores_dict):
    return pd.DataFrame(scores_dict)

# Create a function to get the buy and sell signals
def implement_bb_strategy(predictions, rate, model_name):
    """
    implement the bb strategy (bollinger bands) on the model predictions
    """
    buy_signal = [] #buy list
    sell_signal = [] #sell list
    hold_signal = [] # hold list
   
    df = pd.DataFrame(predictions, columns=['predictions'])
    df['rolling_avg'] = df['predictions'].rolling(rate).mean()
    df['rolling_std'] = df['predictions'].rolling(rate).std()
    df['Upper Band']  = df['rolling_avg'] + (df['rolling_std'] * 1)
    df['Lower Band']  = df['rolling_avg'] - (df['rolling_std'] * 1)
    df = df.dropna()
    df = df.reset_index(drop = True)
    for i in range(len(df['predictions'])):
        if df['predictions'].values[i] > df['Upper Band'].values[i]: #Then you should sell 
            buy_signal.append(np.nan)
            sell_signal.append(df['predictions'][i])
            hold_signal.append(np.nan)
        elif df['predictions'].values[i] < df['Lower Band'].values[i]: #Then you should buy
            sell_signal.append(np.nan)
            buy_signal.append(df['predictions'][i])
            hold_signal.append(np.nan)
        else:
            buy_signal.append(np.nan)
            sell_signal.append(np.nan)
            hold_signal.append(df['predictions'][i]) #Then you should hold
            
    labels_df = pd.DataFrame(list(zip(buy_signal, sell_signal, hold_signal)), columns=['buy_signal', 'sell_signal', 'hold_signal'])
    labels_df.to_csv(f'output/predictions/{model_name}_predictions.csv')

            
    fig, ax = plt.subplots(figsize=(16, 5))
    df['predictions'].plot(label = 'PREDICTED PRICES', linewidth=1.5)
    df['Upper Band'].plot(label = 'UPPER BB', linestyle = '--', linewidth = 1.5)
    df['rolling_avg'].plot(label = 'MIDDLE BB', linestyle = '--', linewidth = 1.5)
    df['Lower Band'].plot(label = 'LOWER BB', linestyle = '--', linewidth = 1.5)
    
    plt.scatter(df.index, buy_signal, marker = '^', color = 'green', label = 'BUY', s = 100)
    plt.scatter(df.index, np.absolute(sell_signal), marker = 'v', color = 'red', label = 'SELL', s = 100)
    plt.scatter(df.index, hold_signal, marker = '*', color = 'blue', label = 'HOLD', s = 100)

    plt.title(f'Bollinger Bands Strategy Trading Signals for {model_name}', fontsize=20)
    plt.legend(loc = 'upper left')
    plt.show()
    return labels_df

def percentage(df):
    percentage_list = []
    index_list = [0]
    for i in range(len(df)):
        if df.buy_signal[i] == 1 or df.sell_signal[i] == 1:
            index_list.append(i)
    index_list.append(df.shape[0]-1)

    for i in range(len(df)):
        if len(df[df.buy_signal.astype(str).str.contains("1") == True]) != 0 or len(df[df.sell_signal.astype(str).str.contains("1") == True]) != 0:
            if df.buy_signal[i] == 1 or df.sell_signal[i] == 1: 
                for j in range (len(index_list)):
                    if j == len(index_list)-1:
                        break

                    if j == len(index_list) -3:
                        start = index_list[j+1]
                        end = index_list[-1]
                    else:
                        start = index_list[j]
                        end = index_list[j+1]
                        if j == len(index_list) -2:
                            break
                    difference = df.price[end] - df.price[start]
                    percent = (difference/df.price[start]) * 100
                    percentage_list.append(percent)

                break


        else:
            difference = df.price[df.shape[0]-1] - df.price[0]
            percent = (difference/df.price[0]) * 100
            percentage_list.append(percent)
            break
    
    return (f'The saved percentage of money by following the model recommendations is: {percentage_list}')




def stat_ml_models(df, data_name, X_train, y_train, X_test, y_test, scores_dict):
    """
    return arima, sarima, arimax and prophet models results
    """
    arima_model_name, arima_mae, arima_y_pred = arima_model(data_name, y_train, y_test)
    scores_dict['Model'].append(arima_model_name)
    scores_dict['Test Score'].append(arima_mae)

    sarima_model_name, sarima_mae, sarima_y_pred = sarima_model(data_name, y_train, y_test)
    scores_dict['Model'].append(sarima_model_name)
    scores_dict['Test Score'].append(sarima_mae)

    arimax_model_name, arimax_mae, arimax_y_pred = arimax_model(data_name, X_train, y_train, X_test, y_test)
    scores_dict['Model'].append(arimax_model_name)
    scores_dict['Test Score'].append(arimax_mae)

    return arima_model_name, arima_y_pred, sarima_model_name, sarima_y_pred, arimax_model_name, arimax_y_pred

def univariate_LSTM(data_name, y_train, y_test, scores_dict):
    """
    return the univariate LSTM moled result
    """
    univariate_X_train, univariate_y_train =  univariate_Lookback_data(y_train, 6)
    univariate_X_test, univariate_y_test =  univariate_Lookback_data(y_test, 6)

    ratio = len(univariate_X_train) * 0.9
    univariate_X_val = univariate_X_train.iloc[int(ratio):]
    univariate_X_train = univariate_X_train.iloc[:int(ratio)]


    univariate_y_val = univariate_y_train.iloc[int(ratio):]
    univariate_y_train = univariate_y_train.iloc[:int(ratio)]

    univariate_LSTM_model_name, univariate_LSTM_model_mae, univariate_LSTM_y_pred = LSTM(univariate_X_train, univariate_y_train, univariate_X_val, univariate_y_val, univariate_X_test, univariate_y_test, 1, data_name)    
    scores_dict['Model'].append(univariate_LSTM_model_name)
    scores_dict['Test Score'].append(univariate_LSTM_model_mae)
    
    return univariate_LSTM_model_name, univariate_LSTM_y_pred


def multivariate_LSTM(data_name, X_train, y_train, X_test, y_test, scores_dict):
    """
    return the multivariate LSTM moled result
    """    
    ratio = len(X_train) * 0.9
    multivariate_X_val = X_train.iloc[int(ratio):]
    multivariate_X_train = X_train.iloc[:int(ratio)]

    multivariate_y_val = y_train.iloc[int(ratio):]
    multivariate_y_train = y_train.iloc[:int(ratio)]


    multivariate_LSTM_X_train, multivariate_LSTM_y_train =  multivariate_Lookback_data(multivariate_X_train.to_numpy(), multivariate_y_train.to_numpy(),6)
    multivariate_LSTM_X_val, multivariate_LSTM_y_val =  multivariate_Lookback_data(multivariate_X_val.to_numpy(), multivariate_y_val.to_numpy(),6)
    multivariate_LSTM_X_test, multivariate_LSTM_y_test =  multivariate_Lookback_data(X_test.to_numpy(), y_test.to_numpy(), 6)

    multivariate_LSTM_model_name, multivariate_LSTM_model_mae, multivariate_LSTM_y_pred = LSTM(multivariate_LSTM_X_test, multivariate_LSTM_y_test, multivariate_LSTM_X_val, multivariate_LSTM_y_val, multivariate_LSTM_X_test, multivariate_LSTM_y_test, 5, data_name, univariate=False)    
    scores_dict['Model'].append(multivariate_LSTM_model_name)
    scores_dict['Test Score'].append(multivariate_LSTM_model_mae)
    
    return multivariate_LSTM_model_name, multivariate_LSTM_y_pred
    
    
def univariate_ConvLSTM(data_name, y_train, y_test, scores_dict):
    """
    return the univariate ConvLSTM  moled result
    """      
    univariate_X_train, univariate_y_train =  univariate_Lookback_data(y_train, 6)
    univariate_X_test, univariate_y_test =  univariate_Lookback_data(y_test, 6)

    ratio = len(univariate_X_train) * 0.9
    univariate_X_val = univariate_X_train.iloc[int(ratio):]
    univariate_X_train = univariate_X_train.iloc[:int(ratio)]


    univariate_y_val = univariate_y_train.iloc[int(ratio):]
    univariate_y_train = univariate_y_train.iloc[:int(ratio)]
    
    univariate_X_train = univariate_X_train.to_numpy().reshape(univariate_X_train.shape[0], 3, 1, 2, 1)
    univariate_X_val = univariate_X_val.to_numpy().reshape(univariate_X_val.shape[0], 3, 1, 2, 1)
    univariate_X_test = univariate_X_test.to_numpy().reshape(univariate_X_test.shape[0], 3, 1, 2, 1)


    univariate_conv_model_name, univariate_conv_model_mae, univariate_conv_y_pred = ConvLSTM(univariate_X_train,
                         univariate_y_train, 
           univariate_X_val, univariate_y_val, univariate_X_test,univariate_y_test, 
                                                  data_name, input_shape = (3, 1, 2, 1), multivariate=False)
    
    
    scores_dict['Model'].append(univariate_conv_model_name)
    scores_dict['Test Score'].append(univariate_conv_model_mae)
    return univariate_conv_model_name, univariate_conv_y_pred

def multivariate_ConvLSTM(data_name, X_train, y_train, X_test, y_test, scores_dict):
    """
    return the multivariate ConvLSTM  moled result
    """      
    
    ratio = len(X_train) * 0.9
    multivariate_X_val = X_train.iloc[int(ratio):]
    multivariate_X_train = X_train.iloc[:int(ratio)]

    multivariate_y_val = y_train.iloc[int(ratio):]
    multivariate_y_train = y_train.iloc[:int(ratio)]
    
    multivariate_conv_X_train, multivariate_conv_y_train =  multivariate_Lookback_data(tf.convert_to_tensor(multivariate_X_train, dtype=tf.float32), tf.convert_to_tensor(multivariate_y_train, dtype=tf.float32),6)
    multivariate_conv_X_train = multivariate_conv_X_train.reshape(multivariate_conv_X_train.shape[0], 3, 1, 2, 5)

    
    multivariate_conv_X_val, multivariate_conv_y_val =  multivariate_Lookback_data(tf.convert_to_tensor(multivariate_X_val, dtype=tf.float32), tf.convert_to_tensor(multivariate_y_val, dtype=tf.float32),6)
    multivariate_conv_X_val = multivariate_conv_X_val.reshape(multivariate_conv_X_val.shape[0], 3, 1, 2, 5)

     
    multivariate_conv_X_test, multivariate_conv_y_test =  multivariate_Lookback_data(tf.convert_to_tensor(X_test, dtype=tf.float32), tf.convert_to_tensor(y_test, dtype=tf.float32),6)
    multivariate_conv_X_test = multivariate_conv_X_test.reshape(multivariate_conv_X_test.shape[0], 3, 1, 2, 5)

    
    
    
    multivariate_ConvLSTM_model_name, multivariate_ConvLSTM_model_mae, multivariate_ConvLSTM_y_pred = ConvLSTM(multivariate_conv_X_train, 
               multivariate_conv_y_train, multivariate_conv_X_val, multivariate_conv_y_val, multivariate_conv_X_test, 
                                       multivariate_conv_y_test, data_name, input_shape = (3, 1, 2, 5), multivariate=True)

    scores_dict['Model'].append(multivariate_ConvLSTM_model_name)
    scores_dict['Test Score'].append(multivariate_ConvLSTM_model_mae) 
    return multivariate_ConvLSTM_model_name, multivariate_ConvLSTM_y_pred

    
    
def models_report(df, data_name):
    # 1) use the train_test_split function to make the train/test split
    X_train, y_train, X_test, y_test = train_test_split(df)
    X_train = X_train[['Open', 'High', 'Low', 'Vol.', 'Change %']]
    X_test = X_test[['Open', 'High', 'Low', 'Vol.', 'Change %']]

    #Create a scoring data frame and functiont to add score
    scores_dict = {"Model":[],"Test Score":[]}
    # 2) use the stat_ml_models funtion to find the predictions for arima, sarima, arimax and prophet models
    arima_model_name, arima_y_pred, sarima_model_name, sarima_y_pred, arimax_model_name, arimax_y_pred = stat_ml_models(df, data_name, X_train, y_train, X_test, y_test, scores_dict)
    
    # 3) use the univariate_LSTM function to find the univariate LSTM model predictions
    univariate_LSTM_model_name, univariate_LSTM_y_pred = univariate_LSTM(data_name, y_train, y_test, scores_dict)
    
    # 4) use the multivariate_LSTM function to find the multivariate LSTM model predictions
    multivariate_LSTM_model_name, multivariate_LSTM_y_pred = multivariate_LSTM(data_name, X_train, y_train, X_test, y_test, scores_dict)
    # 5) use the univariate_ConvLSTM function to find the univariate ConvLSTM model predictions
    univariate_conv_model_name, univariate_conv_y_pred = univariate_ConvLSTM(data_name, y_train, y_test, scores_dict)
    # 6) use the multivariate_ConvLSTM function to find the multivariate LSTM model predictions
    multivariate_ConvLSTM_model_name, multivariate_ConvLSTM_y_pred = multivariate_ConvLSTM(data_name, X_train, y_train, X_test, y_test, scores_dict)

    # 7) collect all the predictions from the models in one dict
    predictions_dict = {arima_model_name : arima_y_pred,
                        sarima_model_name : sarima_y_pred,
                        arimax_model_name : arimax_y_pred,
                        univariate_LSTM_model_name : univariate_LSTM_y_pred,
                        univariate_conv_model_name : univariate_conv_y_pred,
                        multivariate_LSTM_model_name : multivariate_LSTM_y_pred,
                        multivariate_LSTM_model_name : multivariate_ConvLSTM_y_pred
                       }
    # 8) transform the dict to df
    scores_dict = scores(scores_dict)
    # 9) find the min score
    min_score = scores_dict['Model'][scores_dict['Test Score'].idxmin()]
    # 10) save the min score model prediction into bb_predictions var
    for key in predictions_dict:
        if key == min_score:
            bb_predictions = predictions_dict[key]
            
    
    # 11) implement the bb strategy
    labels_df = implement_bb_strategy(list(bb_predictions), 20, min_score) 
    labels_df = labels_df.fillna(0)
    labels_df.loc[labels_df.buy_signal != 0, "buy_signal"] = 1
    labels_df.loc[labels_df.sell_signal != 0, "sell_signal"] = 1
    labels_df.loc[labels_df.hold_signal != 0, "hold_signal"] = 1
    features = X_test[['Open', 'High', 'Low', 'Vol.', 'Change %']]
    features['price'] = y_test
    predictions_df = features.tail(labels_df.shape[0])
    predictions_df = predictions_df.reset_index(drop=True)
    predictions_with_label_df = pd.concat([predictions_df, labels_df], axis=1)
    predictions_with_label_df.to_csv(f"output/predictions with labels/{data_name}_predictions_with_label_df.csv")
    saved_money = percentage(predictions_with_label_df)
    print(saved_money)
    return scores_dict