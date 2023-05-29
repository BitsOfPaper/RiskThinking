# -*- coding: utf-8 -*-
"""stock_etf_data_pipeline
"""

import os
import pandas as pd
import pyarrow
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from sklearn import metrics
from joblib import dump, load
import logging


if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom

@data_loader
def load_data(*args, **kwargs):
    # Load Kaggle API credentials from environment variables
    kaggle_username = os.environ['KAGGLE_USERNAME']
    kaggle_key = os.environ['KAGGLE_KEY']

    # Authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download a Kaggle dataset
    dataset_name = 'jacksoncrow/stock-market-dataset'
    download_path = 'RiskT'
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    api.dataset_download_files(dataset_name, path=download_path, unzip=True)


    # Verify total files
    stock_files = os.listdir('RiskT/stocks')
    etf_files = os.listdir('RiskT/etfs')
    print(f'Total stock and etf files', len(stock_files) + len(etf_files))


    # Create an empty DataFrame with specified columns and data types
    data = pd.DataFrame({
        'Symbol': pd.Series(dtype='str'),
        'Security Name': pd.Series(dtype='str'),
        'Date': pd.Series(dtype='str'),
        'Open': pd.Series(dtype='float'),
        'High': pd.Series(dtype='float'),
        'Low': pd.Series(dtype='float'),
        'Close': pd.Series(dtype='float'),
        'Adj Close': pd.Series(dtype='float'),
        'Volume': pd.Series(dtype='int64')
    })

    # Print the data structure
    print(data)
    if data.empty:
        print("DataFrame is empty")
    else:
        print("DataFrame is not empty")


    symbols = pd.read_csv('RiskT/symbols_valid_meta.csv')
    i = 0

    for dirname, _, filenames in os.walk('RiskT'):
        for filename in filenames:
            data_path=os.path.join(dirname, filename)
            
            if data_path != 'RiskT/symbols_valid_meta.csv':
                file = pd.read_csv(data_path, na_values=["null"], parse_dates=True)
                file['Symbol'] = filename.replace('.csv','')
                file= pd.merge(file, symbols[['Security Name','Symbol']], how='left', left_on='Symbol', right_on='Symbol')
                if data.empty:
                  data = file
                else:
                  data = pd.concat([data, file])


    data.isnull().sum()
    null_rows = data[data['Security Name'].isnull()]
    null_rows['Symbol'].unique()

    data = data[data['Symbol'] != 'UTX#']
    data.loc[data['Symbol'] == 'AGM-A', 'Security Name'] = 'Federal Agricultural Mortgage Corporation'
    data.loc[data['Symbol'] == 'CARR#', 'Security Name'] = 'Carrier Global Corporation'


    data = data.dropna(subset=['Volume'])
    data.drop_duplicates(inplace=True)
    data['Volume'].astype(int)

    new_order = ['Symbol', 'Security Name', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    data = data.reindex(columns=new_order).sort_values(['Symbol', 'Date'], ascending=[True, True])


    # Save cleaned data as parquet file in stocks_etfs_cleaned
    des_path = 'stocks_etfs_cleaned'
    if not os.path.exists(des_path):
        os.makedirs(des_path)

    data.to_parquet('stocks_etfs_cleaned/stocks_etfs_cleaned.parquet')

    return data

@transformer
def transform(data, *args, **kwargs):
    # Calculate the moving average of the trading volume (Volume) of 30 days per each stock and ETF
    data['vol_moving_avg'] = data.groupby('Symbol')['Volume'].transform(lambda x: x.rolling(30).mean())

    # Similarly, calculate the rolling median for Adj Close
    data['adj_close_rolling_med'] = data.groupby('Symbol')['Adj Close'].transform(lambda x: x.rolling(30).median())

    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)



    # Save transformed data as parquet file in stocks_etfs_transformed
    des_path = 'stocks_etfs_transformed'
    if not os.path.exists(des_path):
        os.makedirs(des_path)

    data.to_parquet('stocks_etfs_transformed/stocks_etfs_transformed.parquet')

    return data

@custom
def transform_custom(data, *args, **kwargs):
    # Set the percentage of rows to select for training
    percentage = 0.8

    grouped = data.groupby('Symbol')

    # Get the number of rows in each group
    n_rows = grouped.size()

    # Calculate the number of rows to select from each group
    n_rows_to_select = (n_rows * percentage).astype(int)

    # Select a percent of rows from each group
    train = grouped.apply(lambda x: x.head(n_rows_to_select[x.name])).reset_index(drop=True)
    test = data.drop(train.index)

    train = train.sort_values(['Symbol', 'Date'], ascending=[True, True])
    test = test.sort_values(['Symbol', 'Date'], ascending=[True, True])

    # Select features and target
    features = ['vol_moving_avg', 'adj_close_rolling_med']
    target = 'Volume'

    # For scaling data
    scaler = MinMaxScaler()

    # Split data into train and test sets
    X_train = scaler.fit_transform(train[features])
    X_test = scaler.transform(test[features])

    y_train = train[target]
    y_test = test[target]

    #import lightgbm as lgb
    # Create LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)

    # Set hyperparameters
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.005
    }

    # Train the model
    num_rounds = 500
    model = lgb.train(params, train_data, num_rounds)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    model_path = 'model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    dump(model, 'model/model.joblib')
    dump(scaler, 'model/scaler.joblib')

    logging.basicConfig(filename='training_logs.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', mode='a')

    logging.info(f'Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred)}')
    logging.info(f'Root Mean Squared Error: {metrics.mean_squared_error(y_test, y_pred)**(1/2)}')
    logging.info(f'R2: {metrics.r2_score(y_test, y_pred)**(1/2)}')

    return None


# Step 1: Raw Data Processing
data = load_data()

# Step 2: Feature Engineering
transformed_data = transform(data)

# Step 3: Model training
transform_custom(transformed_data)