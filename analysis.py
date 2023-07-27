import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def preprocessing(data_name, sheet_num):
    """
    param data_name: The title of the data that will be used in any figure title.
    param sheet_num: The sheet number; the excel file contains many sheets.
    
    This function will read the data in the file (dataset.xlsx) and return a clean df.
    """
    xls = pd.ExcelFile('dataset.xlsx')
    df = xls.parse(sheet_num)
    print('The first 5 rows of the {f_data_name} data set: \n\n{f_data}\n'.format(f_data_name=data_name, f_data=df.head()))
    print('############################################################################')
    df.drop([df.shape[0]-1], axis=0, inplace=True)

    
    # Fixing the Vol columns by converting the k-values to M by dividing by 1000 and convert the column data type into int
    k_data = df[df['Vol.'].astype(str).str.contains('K')]
    df = df[df["Vol."].str.contains("K") == False]
    k_data['Vol.'] = k_data['Vol.'].str.replace('K', '')
    k_data['Vol.'] = k_data['Vol.'].apply(pd.to_numeric)
    k_data['Vol.'] = [(i/1000) for i in k_data['Vol.']]
    df = pd.concat([df, k_data], join="inner")

    
    df['Vol.'] = df['Vol.'].str.replace('M', '').replace('-', '')
    df[["Date"]] = df[["Date"]].apply(pd.to_datetime)
    df[["Price", "Open", "High", "Low", "Change %", 'Vol.']] = df[["Price", "Open", "High", "Low", "Change %", 'Vol.']].apply(pd.to_numeric)
    df = df.sort_values(by=['Date'], ascending=True)
    
    return df, data_name

def stationary(df):
    """
    This function makes the time series stationary.
    """
    # ADF Test to fins the p-value
    result = adfuller(df.Price.values, autolag='AIC')
    if result[1] > 0.05:
        print(f'The original price values: \nThe time series is not stationary and the p-value is {result[1]}')
        result = adfuller(np.diff(df.Price.values), autolag='AIC')
    if result[1] < 0.05:
        print(f'The price values after difference: \nThe time series is stationary and the p-value is {result[1]}')
        difference = df.Price.diff()
        df['difference'] = difference
    else:
        print('Your time series is not stationary, you may need to make another difference')
    return df

def add_exogenous_features(df):
    """
    This function will make new features from the chosen columns by using the mead, std for 3, 7, 30 days. 
    And also other features like day, month.
    """
    df.reset_index(drop=True, inplace=True)
    # the chosen columns
    
    lag_features = ['Open', 'High', 'Low', 'Vol.', 'Change %']
    
    # choose the mean, std days
    window1 = 3
    window2 = 7
    window3 = 30
    
    # rolling
    df_rolled_3d = df[lag_features].rolling(window=window1, min_periods=0)
    df_rolled_7d = df[lag_features].rolling(window=window2, min_periods=0)
    df_rolled_30d = df[lag_features].rolling(window=window3, min_periods=0)
    
    # find the mean
    df_mean_3d = df_rolled_3d.mean().shift(1).reset_index()
    df_mean_7d = df_rolled_7d.mean().shift(1).reset_index()
    df_mean_30d = df_rolled_30d.mean().shift(1).reset_index()

    # fins the std
    df_std_3d = df_rolled_3d.std().shift(1).reset_index()
    df_std_7d = df_rolled_7d.std().shift(1).reset_index()
    df_std_30d = df_rolled_30d.std().shift(1).reset_index()

    # add the features to df
    for feature in lag_features:
        df[f"{feature}_mean_lag{window1}"] = df_mean_3d[feature]
        df[f"{feature}_mean_lag{window2}"] = df_mean_7d[feature]
        df[f"{feature}_mean_lag{window3}"] = df_mean_30d[feature]

        df[f"{feature}_std_lag{window1}"] = df_std_3d[feature]
        df[f"{feature}_std_lag{window2}"] = df_std_7d[feature]
        df[f"{feature}_std_lag{window3}"] = df_std_30d[feature]
    
    # remove the nulls
    df.fillna(df.mean(), inplace=True)
    df.set_index("Date", drop=False, inplace=True)
    
    # add the other features
    df["month"] = df.Date.dt.month
    df["week"] = df.Date.dt.isocalendar().week
    df["day"] = df.Date.dt.day
    df["day_of_week"] = df.Date.dt.dayofweek
    return df

def analysing(df, data_name):
    """
    param df: take a preprocessed d f==> the result from the (preprocessing) function.
    param date name: used to naming the title figures.
    
    This function take a preprocessed data and returnthe df with plotting 4 main figures, and they are:
    1) The Price with it's peaks and troughs.
    2) The yearly and monthly box-plot.
    3) Stationary plot.
    4) Plot the trand, seasonalty, ...ect.
    """
    # 1) Figure 1:
    
    # Get the Peaks and Troughs
    data = df['Price'].values
    doublediff = np.diff(np.sign(np.diff(data)))
    peak_locations = np.where(doublediff == -2)[0] + 1
    doublediff2 = np.diff(np.sign(np.diff(-1*data)))
    trough_locations = np.where(doublediff2 == -2)[0] + 1
    
    # plot the price
    plt.figure(figsize=(20,8))
    plt.plot(df["Price"], linewidth=3, label=f'{data_name} Price')
    # plot the peaks
    plt.scatter(df.Date[peak_locations], df.Price[peak_locations], marker=mpl.markers.CARETUPBASE, color='tab:green', s=100, label='Peaks')
    # plot the Troughs
    plt.scatter(df.Date[trough_locations], df.Price[trough_locations], marker=mpl.markers.CARETDOWNBASE, color='tab:red', s=100, label='Troughs')
    plt.title(f'{data_name} Price history', fontsize=20)
    # adjust the legend
    plt.legend(loc='upper left')
     
    # 2) Figure 2:
    
    # make a new columns for the months and the year
    df['year'] = [d.year for d in df.Date]
    years = df['year'].unique()

    # Draw Plot
    fig, axes = plt.subplots(1, 2, figsize=(20,8), dpi= 80)
    sns.boxplot(x='year', y='Price', data=df, ax=axes[0])
    sns.boxplot(x='month', y='Price', data=df)

    # Set Title
    axes[0].set_title(f'Year-wise Box Plot ({data_name})', fontsize=20); 
    axes[1].set_title(f'Month-wise Box Plot ({data_name})', fontsize=20)
    plt.show()
    
    plt.figure(figsize=(20,8))
    sns.boxplot(x='month', y='Price', data=df, hue='year')
    plt.title(f'Month-wise Box Plot Per Year ({data_name})', fontsize=20)
#     ax = sns.boxplot(x="day", y="total_bill", hue="smoker",
#                  data=tips, palette="Set3")
    
    # 3) Figure 3:
    fig, axes = plt.subplots(1,2, figsize=(20,8))
    pd.Series(df.Price.values).plot(title=f'Not-Stationary {data_name} Price', color='r', ax=axes[0], linewidth=2)
    axes[0].title.set_size(20)
    pd.Series(df.difference.values).plot(title=f'Stationary {data_name} Price', color='b', ax=axes[1], linewidth=2)
    axes[1].title.set_size(20)
    # Additive Decomposition
#     result_add = seasonal_decompose(df.difference, model='additive', extrapolate_trend='freq', period=5)
#     # Plot
#     plt.rcParams.update({'figure.figsize': (22,15)})
#     result_add.plot()
#     plt.show()

    # 4) Figure 4:
    
    # Draw Plot
#     fig, axes = plt.subplots(1,2,figsize=(16,5))
#     plot_acf(df.difference, ax=axes[0])
#     plot_pacf(df.difference, ax=axes[1]);

    return df


def analysis_report(data_name, sheet_num):
    """
    param data_name: The title of the data that will be used in any figure title.
    param sheet_num: The sheet number; the excel file contains many sheets.
    This function will do 4 things:
    1) preprocessing the data.
    2) check if the time series is stationary or not.
    3) add new features based on the existing ones to use them later in the modelling part.
    4) plot some features and stat about the data.
    """
    df, data_name = preprocessing(data_name, sheet_num)
    df = stationary(df)
    df = add_exogenous_features(df)
    df = analysing(df, data_name)
    return df, data_name