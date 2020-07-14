import gc
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.optimize import differential_evolution
import os
from tqdm import tqdm
from utils import save_pickle

# silence warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

print('Loading data...')
BIAS_FOLDER = './data/biases'

# create folder for biases
if not os.path.exists(BIAS_FOLDER):
    os.makedirs(BIAS_FOLDER)
    
# load competiton data
ids_df = pd.read_csv('./data/raw/reveal_ID_site2.csv')
fnc_df = pd.read_csv('./data/raw/fnc.csv')
loading_df = pd.read_csv('./data/raw/loading.csv')
labels_df = pd.read_csv('./data/raw/train_scores.csv')

# load created features
pca_df = pd.read_csv('./data/features/200pca_feats/200pca_3d_k0.csv')
for i in range(1, 6):
    part = pd.read_csv('./data/features/200pca_feats/200pca_3d_k{}.csv'.format(i)); del part['Id']
    pca_df = pd.concat((pca_df, part), axis=1)

# merge data
df = fnc_df.merge(loading_df, on='Id')
df = df.merge(pca_df, how='left', on='Id')
df = df.merge(labels_df, how='left', on='Id')

# split train, test and test site2
df.loc[df['Id'].isin(labels_df['Id']), 'is_test'] = 0
df.loc[~df['Id'].isin(labels_df['Id']), 'is_test'] = 1

train = df.query('is_test==0'); del train['is_test']
test = df.query('is_test==1'); del test['is_test']

test_site = test[test['Id'].isin(ids_df['Id'])]
del train['Id'], test['Id'], test_site['Id']

ic_cols = sorted([c for c in train.columns if c.startswith('IC_')])
fnc_cols = sorted(fnc_df.columns[1:])
pca_cols = sorted(pca_df.columns[1:])

# I. IC cols: test
print('Computing biases for IC cols test...')
bias = {}
for ic in tqdm(ic_cols):
    
    # have to redefine to enable multiprocessing
    def compute_test(b):
        return ks_2samp(train[ic], test[ic] + b[0])[0]
    
    res = differential_evolution(compute_test, [(-0.01, 0.01)], seed=0, workers=1)
    bias[ic] = res.x[0]

# save results
save_pickle(bias, BIAS_FOLDER + '/ic_biases')

# II. IC cols: test-site2
print('Computing biases for IC cols test-site2...')
bias = {}
for ic in tqdm(ic_cols):
    
    # have to redefine to enable multiprocessing
    def compute_test(b):
        return ks_2samp(train[ic], test_site[ic] + b[0])[0]
    
    res = differential_evolution(compute_test, [(-0.01, 0.01)], seed=0, workers=1)
    bias[ic] = res.x[0]

# save results
save_pickle(bias, BIAS_FOLDER + '/ic_biases_site')

# III. FNC cols: test
print('Computing biases for FNC cols test...')
bias = {}
for fnc in tqdm(fnc_cols):
    
    # have to redefine to enable multiprocessing
    def compute_test(b):
        return ks_2samp(train[fnc], test[fnc] + b[0])[0]
    
    res = differential_evolution(compute_test, [(-1, 1)], seed=0, workers=1)
    bias[fnc] = res.x[0]

# save results
save_pickle(bias, BIAS_FOLDER + '/fnc_biases')

# IV. FNC cols: test-site2
print('Computing biases for FNC cols test-site2...')
bias = {}
for fnc in tqdm(fnc_cols):
    
    # have to redefine to enable multiprocessing
    def compute_test(b):
        return ks_2samp(train[fnc], test_site[fnc] + b[0])[0]
    
    res = differential_evolution(compute_test, [(-1, 1)], seed=0, workers=4)
    bias[fnc] = res.x[0]

# save results
save_pickle(bias, BIAS_FOLDER + '/fnc_biases_site')

# V. PCA cols: test
print('Computing biases for PCA cols test...')
bias = {}
for pca in tqdm(pca_cols):
    
    # have to redefine to enable multiprocessing
    def compute_test(b):
        return ks_2samp(train[pca], test[pca] + b[0])[0]
    
    res = differential_evolution(compute_test, [(-20, 20)], seed=0, workers=-1)
    bias[pca] = res.x[0]

# save results
save_pickle(bias, BIAS_FOLDER + '/200pca_biases')

# VI. PCA cols: test-site2
print('Computing biases for PCA cols test-site2...')
bias = {}
for pca in tqdm(pca_cols):
    
    # have to redefine to enable multiprocessing
    def compute_test(b):
        return ks_2samp(train[pca], test_site[pca] + b[0])[0]
    
    res = differential_evolution(compute_test, [(-20, 20)], seed=0, workers=-1)
    bias[pca] = res.x[0]

# save results
save_pickle(bias, BIAS_FOLDER + '/200pca_biases_site')