import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

def read_pickle(f):
    with open(f, 'rb') as handle:
        return pickle.load(handle)

def save_pickle(f, file_path):
    with open(file_path + '.pickle', 'wb') as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)

def scale_select_data(train, test, df_scale, cols, sc=StandardScaler(), scale_factor=0.1, scale_cols=None):
    score_cols = sorted(list(set(cols)))
    sc.fit(df_scale[score_cols])

    train_scaled = train[score_cols].copy()
    test_scaled = test[score_cols].copy()

    if scale_cols:
        train_scaled.loc[:, scale_cols] *= scale_factor
        test_scaled.loc[:, scale_cols] *= scale_factor

    train_scaled = pd.DataFrame(sc.transform(train_scaled), columns=score_cols)
    test_scaled = pd.DataFrame(sc.transform(test_scaled), columns=score_cols)

    return train_scaled, test_scaled