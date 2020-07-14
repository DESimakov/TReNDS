import os
import gc
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import OrthogonalMatchingPursuit, BayesianRidge, ElasticNet, OrthogonalMatchingPursuitCV, HuberRegressor

from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from rgf.sklearn import RGFRegressor
from sklearn.svm import NuSVR

from tqdm import tqdm

from trends import NMAE, TrendsModelSklearn
from utils import scale_select_data, save_pickle, read_pickle

def run(seed):
    
    # create folders for scores models and preds
    folder_models = './models/age/scores/'
    if not os.path.exists(folder_models):
        os.makedirs(folder_models)

    folder_preds = './predicts/age/scores/'
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
    y = train['age'].copy().reset_index(drop=True)

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
    names = ['RGF', 'ENet', 'BRidge', 'Huber', 'OMP']
    names = [name + '_fnc_seed{}'.format(seed) for name in names]
    pack = [RGFRegressor(max_leaf=1000, reg_depth=5, normalize=True),
    ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=0),
    BayesianRidge(),
    HuberRegressor(epsilon=2.5, alpha=1),
    OrthogonalMatchingPursuit(n_nonzero_coefs=300)]
    
    # train models
    zoo = TrendsModelSklearn(pack, seed=seed)
    zoo.fit([train_for_score]*5, y)
    score_blend = zoo.blend_oof()
    pred = zoo.predict([test_for_score]*5, names)

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
    names = ['RGF', 'ENet', 'BRidge', 'OMP']
    names = [name + '_pca_seed{}'.format(seed) for name in names]
    pack = [RGFRegressor(max_leaf=1000, reg_depth=5, min_samples_leaf=100, normalize=True),
    ElasticNet(alpha=0.2, l1_ratio=0.2, random_state=0),
    BayesianRidge(),
    OrthogonalMatchingPursuit()]

    # train models
    zoo = TrendsModelSklearn(pack, seed=seed)
    zoo.fit([train_for_score]*4, y)
    score_blend = zoo.blend_oof()
    pred = zoo.predict([test_for_score]*4, names)

    # save oof, pred, models
    np.save(folder_preds + 'pca_score_seed{}.npy'.format(seed), score_blend)
    np.save(folder_preds + 'pca_score_test_seed{}.npy'.format(seed), pred)
    zoo.save_models(names, folder=folder_models)

    # IV. Create im score
    print('Creating IM score...')

    # prepare datasets for pca score
    train_for_score, test_for_score = scale_select_data(train, test, df_scale, im_cols)

    # define models
    names = ['RGF', 'ENet', 'BRidge', 'OMP']
    names = [name + '_im_seed{}'.format(seed) for name in names]
    pack = [RGFRegressor(max_leaf=1000, reg_depth=5, min_samples_leaf=100, normalize=True),
    ElasticNet(alpha=0.2, l1_ratio=0.2, random_state=0),
    BayesianRidge(),
    OrthogonalMatchingPursuit()]
    
    # train models
    zoo = TrendsModelSklearn(pack, seed=seed)
    zoo.fit([train_for_score]*4, y)
    score_blend = zoo.blend_oof()
    pred = zoo.predict([test_for_score]*4, names)

    # save oof, pred, models
    np.save(folder_preds + 'im_score_seed{}.npy'.format(seed), score_blend)
    np.save(folder_preds + 'im_score_test_seed{}.npy'.format(seed), pred)
    zoo.save_models(names, folder=folder_models)

    # V. Create dl score
    print('Creating DL score...')

    # prepare datasets for pca score
    train_for_score, test_for_score = scale_select_data(train, test, df_scale, dl_cols)

    # define models
    names = ['RGF', 'ENet', 'BRidge']
    names = [name + '_dl_seed{}'.format(seed) for name in names]
    pack = [RGFRegressor(max_leaf=1000, reg_depth=5, min_samples_leaf=100, normalize=True),
    ElasticNet(alpha=0.2, l1_ratio=0.2, random_state=0),
    BayesianRidge()]

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
    print('Reading scores from ', folder_preds)

    # add scores
    for prefix in ['fnc', 'agg', 'im', 'pca', 'dl']:
        train[prefix + '_score'] = np.load(folder_preds + '{}_score_seed{}.npy'.format(prefix, seed))
        test[prefix + '_score'] = np.load(folder_preds + '{}_score_test_seed{}.npy'.format(prefix, seed))
    score_cols = [c for c in train.columns if c.endswith('_score')]

    # save df for scaling
    df_scale = pd.concat([train, test], axis=0)

    # create differents datasets
    # linear
    linear_cols = sorted(list(set(ic_cols + fnc_cols + pca_cols + agg_cols + im_cols) - set(['IC_20'])))
    train_linear, test_linear = scale_select_data(train, test, df_scale, linear_cols)

    # kernel
    kernel_cols = sorted(list(set(ic_cols + pca_cols) - set(['IC_20'])))
    train_kernel, test_kernel = scale_select_data(train=train, test=test, df_scale=df_scale, cols=kernel_cols, scale_cols=pca_cols)

    # score
    sc_cols = sorted(list(set(ic_cols + score_cols) - set(['IC_20'])))
    train_sc, test_sc = scale_select_data(train, test, df_scale, sc_cols)

    # dl
    dict_cols = sorted(list(set(ic_cols + fnc_cols + dl_cols + im_cols + agg_cols) - set(['IC_20'])))
    train_dl, test_dl = scale_select_data(train, test, df_scale, dict_cols)

    # learning process on different datasets
    names = ['MLP', 'RGF', 'SVM', 'BR', 'OMP', 'EN', 'KR']
    names = [name + '_seed{}'.format(seed) for name in names]
    pack = [MLPRegressor(activation='tanh', random_state=0),
    RGFRegressor(max_leaf=1500, loss='Abs'),
    NuSVR(C=10, nu=0.4, kernel='rbf'),
    BayesianRidge(),
    OrthogonalMatchingPursuitCV(),
    ElasticNet(alpha=0.5, l1_ratio=0.7, random_state=0),
    KernelRidge(kernel='poly', alpha=0.5)]

    zoo = TrendsModelSklearn(pack, seed=seed)
    zoo.fit([train_sc]*2 + [train_kernel] + [train_linear]*2 + [train_dl]*2, y)
    de_blend = zoo.blend_oof()
    preds = zoo.predict([test_sc]*2 + [test_kernel] + [test_linear]*2 + [test_dl]*2, names, is_blend=False)

    # rewrite folders for models and preds
    folder_models = './models/age/stack/'
    if not os.path.exists(folder_models):
        os.makedirs(folder_models)

    folder_preds = './predicts/age/stack/'
    if not os.path.exists(folder_preds):
        os.makedirs(folder_preds)

    print('Saving models to', folder_models)
    print('Saving predictions to', folder_preds)

    # save oofs and models
    zoo.save_oofs(names, folder=folder_preds)
    zoo.save_models(names, folder=folder_models)

    # stacking predictions
    print('Stacking predictions...')
    folds = KFold(n_splits=10, shuffle=True, random_state=0)
    stack = pd.DataFrame(zoo.oof_preds).T
    stack.columns = names

    model_stacker_rgf = RGFRegressor(max_leaf=1000, reg_depth=25, verbose=False)
    rgf_pred = cross_val_predict(model_stacker_rgf, stack, y.dropna(), cv=folds, n_jobs=-1)

    model_stacker_br = BayesianRidge()
    br_pred = cross_val_predict(model_stacker_br, stack, y.dropna(), cv=folds, n_jobs=-1)

    model_stacker_rgf.fit(stack, y.dropna())
    model_stacker_br.fit(stack, y.dropna())

    # save models
    save_pickle(model_stacker_br, folder_models + 'BRidge_stack_seed{}'.format(seed))
    save_pickle(model_stacker_rgf, folder_models + 'RGF_stack_seed{}'.format(seed))
    print('Final age NMAE: {:.5f}'.format(NMAE(y, 0.75*br_pred + 0.25*rgf_pred)))
    
    test_preds = pd.DataFrame(preds).T
    test_preds.columns = names

    age_prediction = pd.DataFrame()
    age_prediction['Id'] = test['Id'].values
    age_prediction['pred'] = 0.25*model_stacker_rgf.predict(test_preds) + 0.75*model_stacker_br.predict(test_preds)
    age_prediction.to_csv(folder_preds + 'age_stack_seed{}.csv'.format(seed), index=False)
    print('age seed pred is saved as', folder_preds + 'age_stack_seed{}.csv'.format(seed))









