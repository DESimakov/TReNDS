import os
import gc
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, OrthogonalMatchingPursuit, BayesianRidge, ElasticNet, OrthogonalMatchingPursuitCV, HuberRegressor, Lasso

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from rgf.sklearn import RGFRegressor
from sklearn.svm import NuSVR

from tqdm import tqdm

from trends import NMAE, TrendsModelSklearn
from utils import scale_select_data, save_pickle, read_pickle

def run(seed):
    
    # create folders for scores models and preds
    folder_models = './models/domain2_var2/scores/'
    if not os.path.exists(folder_models):
        os.makedirs(folder_models)

    folder_preds = './predicts/domain2_var2/scores/'
    if not os.path.exists(folder_preds):
        os.makedirs(folder_preds)

    print('Loading data...')

    # load biases
    ic_bias = read_pickle('./data/biases/ic_biases.pickle')
    ic_bias_site = read_pickle('./data/biases/ic_biases_site.pickle')
    fnc_bias = read_pickle('./data/biases/fnc_biases.pickle')
    fnc_bias_site = read_pickle('./data/biases/fnc_biases_site.pickle')
    pca_bias = read_pickle('./data/biases/200pca_biases.pickle')
    pca_bias_site = read_pickle('./data/biases/200pca_biases_site.pickle')

    # load classifier and add extra sites2
    extra_site = pd.DataFrame()
    extra_site['Id'] = np.load('./predicts/classifier/site2_test_new_9735.npy')

    # load competiton data
    ids_df = pd.read_csv('./data/raw/reveal_ID_site2.csv')
    fnc_df = pd.read_csv('./data/raw/fnc.csv')
    loading_df = pd.read_csv('./data/raw/loading.csv')
    labels_df = pd.read_csv('./data/raw/train_scores.csv')

    ids_df = ids_df.append(extra_site)
    print('Detected Site2 ids count: ', ids_df['Id'].nunique())

    # load created features
    agg_df = pd.read_csv('./data/features/agg_feats.csv')
    im_df = pd.read_csv('./data/features/im_feats.csv')
    dl_df = pd.read_csv('./data/features/dl_feats.csv')
    
    pca_df = pd.read_csv('./data/features/200pca_feats/200pca_3d_k0.csv')
    for i in range(1, 6):
        part = pd.read_csv('./data/features/200pca_feats/200pca_3d_k{}.csv'.format(i)); del part['Id']
        pca_df = pd.concat((pca_df, part), axis=1)

    # merge data
    ic_cols = list(loading_df.columns[1:])
    fnc_cols = list(fnc_df.columns[1:])
    agg_cols = list(agg_df.columns[1:])
    im_cols = list(im_df.columns[1:])
    pca_cols = list(pca_df.columns[1:])
    dl_cols = list(dl_df.columns[1:])
    pca0_cols = [c for c in pca_cols if 'k0' in c]

    df = fnc_df.merge(loading_df, on='Id')
    df = df.merge(agg_df, how='left', on='Id')
    df = df.merge(im_df, how='left', on='Id')
    df = df.merge(pca_df, how='left', on='Id')
    df = df.merge(dl_df, how='left', on='Id')
    df = df.merge(labels_df, how='left', on='Id')

    del loading_df, fnc_df, agg_df, im_df, pca_df
    gc.collect()

    # split train and test
    df.loc[df['Id'].isin(labels_df['Id']), 'is_test'] = 0
    df.loc[~df['Id'].isin(labels_df['Id']), 'is_test'] = 1

    train = df.query('is_test==0'); del train['is_test']
    test = df.query('is_test==1'); del test['is_test']
    y = train['domain2_var2'].copy().reset_index(drop=True)
    d22_index = list(train['domain2_var2'].dropna().index)

    # apply biases
    for c in ic_bias_site.keys():
        test.loc[~test['Id'].isin(ids_df['Id']), c] += ic_bias[c]
        test.loc[test['Id'].isin(ids_df['Id']), c] += ic_bias_site[c]

    for c in fnc_bias_site.keys():
        test.loc[~test['Id'].isin(ids_df['Id']), c] += fnc_bias[c]
        test.loc[test['Id'].isin(ids_df['Id']), c] += fnc_bias_site[c]
    
    for c in pca_bias_site.keys():
        test.loc[~test['Id'].isin(ids_df['Id']), c] += pca_bias[c]
        test.loc[test['Id'].isin(ids_df['Id']), c] += pca_bias_site[c]

    # save df for scaling
    df_scale = pd.concat([train, test], axis=0)

    # I. Create fnc score
    print('Creating FNC score...')
    
    # prepare datasets for fnc score
    train_for_score, test_for_score = scale_select_data(train, test, df_scale, fnc_cols)
    
    # define models
    names = ['ENet', 'BRidge', 'Huber', 'OMP']
    names = [name + '_fnc_seed{}'.format(seed) for name in names]
    pack = [ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=0),
    BayesianRidge(),
    HuberRegressor(epsilon=2.5, alpha=1),
    OrthogonalMatchingPursuit(n_nonzero_coefs=300)]
    
    # train models
    zoo = TrendsModelSklearn(pack, seed=seed)
    zoo.fit([train_for_score]*4, y)
    score_blend = zoo.blend_oof()
    pred = zoo.predict([test_for_score]*4, names)

    # save oof, pred, models
    np.save(folder_preds + 'fnc_score_seed{}.npy'.format(seed), score_blend)
    np.save(folder_preds + 'fnc_score_test_seed{}.npy'.format(seed), pred)
    zoo.save_models(names, folder=folder_models)

    # II. Create agg score
    print('Creating AGG score...')

    # prepare datasets for agg score
    train_for_score, test_for_score = scale_select_data(train, test, df_scale, agg_cols)

    # define models
    names = ['RGF', 'ENet', 'Huber']
    names = [name + '_agg_seed{}'.format(seed) for name in names]
    pack = [RGFRegressor(max_leaf=1000, reg_depth=5, min_samples_leaf=100, normalize=True),
    ElasticNet(alpha=0.05, l1_ratio=0.3, random_state=0),
    HuberRegressor(epsilon=2.5, alpha=1)]

    # train models
    zoo = TrendsModelSklearn(pack, seed=seed)
    zoo.fit([train_for_score]*3, y)
    score_blend = zoo.blend_oof()
    pred = zoo.predict([test_for_score]*3, names)

    # save oof, pred, models
    np.save(folder_preds + 'agg_score_seed{}.npy'.format(seed), score_blend)
    np.save(folder_preds + 'agg_score_test_seed{}.npy'.format(seed), pred)
    zoo.save_models(names, folder=folder_models)

    # III. Create pca score
    print('Creating PCA score...')

    # prepare datasets for pca score
    train_for_score, test_for_score = scale_select_data(train, test, df_scale, pca_cols)

    # define models
    names = ['ENet', 'BRidge', 'OMP']
    names = [name + '_pca_seed{}'.format(seed) for name in names]
    pack = [ElasticNet(alpha=0.2, l1_ratio=0.2, random_state=0),
    BayesianRidge(),
    OrthogonalMatchingPursuit()]

    # train models
    zoo = TrendsModelSklearn(pack, seed=seed)
    zoo.fit([train_for_score]*3, y)
    score_blend = zoo.blend_oof()
    pred = zoo.predict([test_for_score]*3, names)

    # save oof, pred, models
    np.save(folder_preds + 'pca_score_seed{}.npy'.format(seed), score_blend)
    np.save(folder_preds + 'pca_score_test_seed{}.npy'.format(seed), pred)
    zoo.save_models(names, folder=folder_models)

    # IV. Create im score
    print('Creating IM score...')

    # prepare datasets for pca score
    train_for_score, test_for_score = scale_select_data(train, test, df_scale, im_cols)

    # define models
    names = ['ENet', 'BRidge', 'OMP']
    names = [name + '_im_seed{}'.format(seed) for name in names]
    pack = [ElasticNet(alpha=0.2, l1_ratio=0.2, random_state=0),
    BayesianRidge(),
    OrthogonalMatchingPursuit()]
    
    # train models
    zoo = TrendsModelSklearn(pack, seed=seed)
    zoo.fit([train_for_score]*3, y)
    score_blend = zoo.blend_oof()
    pred = zoo.predict([test_for_score]*3, names)

    # save oof, pred, models
    np.save(folder_preds + 'im_score_seed{}.npy'.format(seed), score_blend)
    np.save(folder_preds + 'im_score_test_seed{}.npy'.format(seed), pred)
    zoo.save_models(names, folder=folder_models)

    # V. Create dl score
    print('Creating DL score...')

    # prepare datasets for pca score
    train_for_score, test_for_score = scale_select_data(train, test, df_scale, dl_cols)

    # define models
    names = ['ENet', 'BRidge', 'OMP']
    names = [name + '_dl_seed{}'.format(seed) for name in names]
    pack = [ElasticNet(alpha=0.2, l1_ratio=0.2, random_state=0),
    BayesianRidge(),
    OrthogonalMatchingPursuit()]

    # train models
    zoo = TrendsModelSklearn(pack, seed=seed)
    zoo.fit([train_for_score]*3, y)
    score_blend = zoo.blend_oof()
    pred = zoo.predict([test_for_score]*3, names)

    # save oof, pred, models
    np.save(folder_preds + 'dl_score_seed{}.npy'.format(seed), score_blend)
    np.save(folder_preds + 'dl_score_test_seed{}.npy'.format(seed), pred)
    zoo.save_models(names, folder=folder_models)

    # VI. Training and predicting procedure
    print('Training has started...')

    # add scores
    for prefix in ['fnc', 'agg', 'im', 'pca', 'dl']:
        train.loc[d22_index, prefix + '_score'] = np.load(folder_preds + '/{}_score_seed{}.npy'.format(prefix, seed))
        test.loc[:, prefix + '_score'] = np.load(folder_preds + '/{}_score_test_seed{}.npy'.format(prefix, seed))
    score_cols = [c for c in train.columns if c.endswith('_score')]

    # save df for scaling
    df_scale = pd.concat([train, test], axis=0)

    # create differents datasets
    # linear
    linear_cols = sorted(list(set(ic_cols + fnc_cols + pca_cols) - set(['IC_20'])))
    train_linear, test_linear = scale_select_data(train, test, df_scale, linear_cols)

    # kernel
    kernel_cols = sorted(list(set(ic_cols + pca_cols) - set(['IC_20'])))
    train_kernel, test_kernel = scale_select_data(train=train, test=test, df_scale=df_scale, cols=kernel_cols, scale_factor=0.1, scale_cols=pca_cols, sc=StandardScaler())

    # score
    sc_cols = sorted(list(set(ic_cols + score_cols) - set(['IC_20'])))
    train_sc, test_sc = scale_select_data(train, test, df_scale, sc_cols)

    # learning process on different datasets
    names = ['GP', 'SVM', 'BR', 'KR']
    names = [name + '_seed{}'.format(seed) for name in names]
    pack = [GaussianProcessRegressor(DotProduct(), random_state=0),
    NuSVR(C=2, kernel='rbf'),
    BayesianRidge(),
    KernelRidge(kernel='poly', alpha=30)]

    zoo = TrendsModelSklearn(pack, seed=seed)
    zoo.fit([train_sc] + [train_kernel] + [train_linear]*2, y)
    de_blend = zoo.blend_oof()
    preds = zoo.predict([test_sc] + [test_kernel] + [test_linear]*2, names, is_blend=True)

    # rewrite folders for models and preds
    folder_models = './models/domain2_var2/stack/'
    if not os.path.exists(folder_models):
        os.makedirs(folder_models)

    folder_preds = './predicts/domain2_var2/stack/'
    if not os.path.exists(folder_preds):
        os.makedirs(folder_preds)

    print('Saving models to', folder_models)
    print('Saving predictions to', folder_preds)

    # save oofs and models
    zoo.save_oofs(names, folder=folder_preds)
    zoo.save_models(names, folder=folder_models)

    # stacking predictions
    print('Stacking predictions...')
    d22_prediction = pd.DataFrame()
    d22_prediction['Id'] = test['Id'].values
    d22_prediction['pred'] = preds
    d22_prediction.to_csv(folder_preds + 'domain2_var2_stack_seed{}.csv'.format(seed), index=False)
    print('domain2_var2 seed pred is saved as', folder_preds + 'domain2_var2_stack_seed{}.csv'.format(seed))






