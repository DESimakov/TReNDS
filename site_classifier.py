import argparse

import numpy as np
import pandas as pd
import os

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from copy import deepcopy
import pickle

def main():  
    parser = argparse.ArgumentParser(description='train site2 classifier and make test prediction')

    parser.add_argument('--data-path', default='./data',
                        help='path to data folder, default ./data')
    parser.add_argument('--path-to-save', default='./predicts/classifier',
                        help='path to save prediction, default ./predicts/classifier')
    parser.add_argument('--path-to-save-model', default='./models/classifier',
                        help='path to save classifier model, default ./models/classifier')
    
    args = parser.parse_args()
    data_path = args.data_path
    path_to_save = args.path_to_save
    path_to_save_model = args.path_to_save_model
    
    for _path in [path_to_save, path_to_save_model]:
        if not os.path.exists(_path):
            os.makedirs(_path)
  
    # prepare data
    ids_df = pd.read_csv(os.path.join(data_path, 'raw/reveal_ID_site2.csv'))
    fnc_df = pd.read_csv(os.path.join(data_path, 'raw/fnc.csv'))
    loading_df = pd.read_csv(os.path.join(data_path, 'raw/loading.csv'))
    labels_df = pd.read_csv(os.path.join(data_path, 'raw/train_scores.csv'))
        
    df = fnc_df.merge(loading_df, on='Id')
    
    df.loc[df['Id'].isin(labels_df['Id']), 'site2'] = 0
    df.loc[~df['Id'].isin(labels_df['Id']), 'site2'] = 1
       
    train_index = np.array(labels_df.Id)
    test_index = np.array(list(set(loading_df.Id) - set(train_index)))
    print('Data loaded')
    
    
    # load PCA
    data_pca = pd.read_csv(os.path.join(data_path, 'features/200pca_feats/200pca_3d_k0.csv'))
    for i in range(1, 6):
        part = pd.read_csv(os.path.join(data_path, 'features/200pca_feats/200pca_3d_k{}.csv'.format(i)))
        del part['Id']
        data_pca = pd.concat((data_pca, part), axis=1)
    
    # merge all data
    df = df.merge(data_pca, how='left', on='Id')
    y = df.pop('site2')
    
    ic_cols = loading_df.columns[1:]
    fnc_cols = fnc_df.columns[1:]
    pca_cols = data_pca.columns[1:]
    
    # simple preprocessing
    sc = StandardScaler()
    data_lr = df[list(ic_cols) + list(fnc_cols) + list(pca_cols)]
    data_lr = sc.fit_transform(data_lr)
    
    
    # train model
    cond = df['Id'].isin(ids_df['Id']) | df['Id'].isin(labels_df['Id'])
    model_lr = ElasticNet(**{'alpha':0.01, 'l1_ratio':0.3, 'random_state':0, 'selection':'random'})
    
    y_train = y.values[cond]
    X_train = data_lr[cond]
    X_test = data_lr[~cond]
    y_test = y.values[~cond]
    
    
    folds = list(KFold(n_splits=5, shuffle=True, random_state=0).split(y_train))
    
    pred_test = np.zeros(len(X_test))
    oof = np.zeros(len(X_train))
    
    models = []
    for n, (tr, te) in enumerate(folds):
        
        model_lr.fit(X_train[tr], y_train[tr])
        cond_test = (model_lr.predict(X_test) > 0.3)
        
        model_lr.fit(np.vstack([X_train[tr], X_test[cond_test]]),
                     np.concatenate([y_train[tr], y_test[cond_test]]))
        
    
        models.append(deepcopy(model_lr))
        oof[te] = model_lr.predict(X_train[te])
        pred_test += model_lr.predict(X_test)
        
    pred_test /= (n+1)   
        
    print(f'auc: {roc_auc_score(y_train, oof)}')
    print(f'mcc: {matthews_corrcoef(y_train, oof > 0.55)}')
    
    _path = os.path.join(path_to_save, 'site2_test_new_9735.npy')
    np.save(_path, test_index[~pd.Series(test_index).isin(ids_df['Id'])][((pred_test) > 0.55)])
    
    
    _path = os.path.join(path_to_save_model, 'classifier_folds_list.pickle')

    with open(_path, 'wb') as f:
        pickle.dump(models, f)
        
    print('All saved')
    
    
        
if __name__ == '__main__':
    main()