import os
import numpy as np
import pandas as pd

def run():

    print('Loading data...')

    # load competiton data
    fnc_df = pd.read_csv('./data/raw/fnc.csv')
    loading_df = pd.read_csv('./data/raw/loading.csv')
    labels_df = pd.read_csv('./data/raw/train_scores.csv')

    # merge data
    df = fnc_df.merge(loading_df, on='Id')
    df = df.merge(labels_df, how='left', on='Id')

    # split train and test
    df.loc[df['Id'].isin(labels_df['Id']), 'is_test'] = 0
    df.loc[~df['Id'].isin(labels_df['Id']), 'is_test'] = 1

    train = df.query('is_test==0')
    test = df.query('is_test==1')
    y_median = train['domain1_var2'].median()

    # folder for pred
    folder_preds = './predicts/domain1_var2/stack/'
    if not os.path.exists(folder_preds):
        os.makedirs(folder_preds)

    # making prediction
    print('Creating prediction...')
    d12_prediction = pd.DataFrame()
    d12_prediction['Id'] = test['Id'].values
    d12_prediction['pred'] = y_median
    d12_prediction.to_csv(folder_preds + 'domain1_var2_stack.csv', index=False)
    print('domain1_var2 pred is saved as', folder_preds + 'domain1_var2_stack.csv')






