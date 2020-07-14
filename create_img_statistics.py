import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse

import h5py
from joblib import Parallel, delayed

def load_subject(subject_filename):

    subject_data = None
    with h5py.File(subject_filename, 'r') as f:
        subject_data = f['SM_feature'][()]
    # It's necessary to reorient the axes, since h5py flips axis order
    subject_data = np.moveaxis(subject_data, [0,1,2,3], [3,2,1,0])
    return subject_data

def create_statistics(i, path):
    '''
    create statistical features from 3d data
    '''
    subject_filename = os.path.join(path, str(i) + '.mat')
    subject_data = load_subject(subject_filename).reshape(-1, 53)
    _std = np.std(subject_data, axis=0)
    _mean = np.mean(subject_data, axis=0)
    _min = np.min(subject_data, axis=0)
    _max = np.max(subject_data, axis=0)
    _q95 = np.quantile(subject_data, 0.95, axis=0)
    _q05 = np.quantile(subject_data, 0.05, axis=0)
    return np.hstack([_std, _mean, _min, _max, _q95, _q05])

def main():  
    parser = argparse.ArgumentParser(description='create statistics from 3d fMRI data')

    parser.add_argument('--data-path', default='./data/raw',
                        help='path to original images with fMRI_train and fMRI_test folders, default ./data/raw')
    parser.add_argument('--path-to-save', default='./data/features',
                        help='path to save features, default ./data/features')
    parser.add_argument('--n-jobs', type=int, default=24,
                        help='number of jobs in multiprocessing. Default 24')
    
    args = parser.parse_args()
    data_path = args.data_path
    path_to_save = args.path_to_save
    n_jobs = args.n_jobs
    
    # load and get train and test id
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        
    loading = pd.read_csv(os.path.join(data_path, 'loading.csv'), index_col = ['Id'])
    y = pd.read_csv(os.path.join(data_path, 'train_scores.csv'), index_col = ['Id'])
    
    train_index = np.array(y.index)
    test_index = np.array(list(set(loading.index) - set(train_index)))
    
    train_index = np.array(y.index)
    test_index = np.array(list(set(loading.index) - set(train_index)))
    print('Data loaded')
    
    # create features

    cols = ['IM_' + str(i) for i in np.arange(318)]
    
    train_im = pd.DataFrame(index=pd.Series(train_index, name='Id'), columns=cols)
    _path = os.path.join(data_path, 'fMRI_train')
    res = Parallel(n_jobs=n_jobs)(delayed(create_statistics)(i, _path) for i in tqdm(train_index))
    train_im.loc[:, cols] = np.vstack(res)
    cond = list(train_im.sum()[(train_im.sum() != 0)].index)
    train_im = train_im.loc[:, cond]
    print('Train feats created')
    
    test_im = pd.DataFrame(index=pd.Series(test_index, name='Id'), columns=cols)
    _path = os.path.join(data_path, 'fMRI_test')
    res = Parallel(n_jobs=n_jobs)(delayed(create_statistics)(i, _path) for i in tqdm(test_index))
    test_im.loc[:, cols] = np.vstack(res)
    test_im = test_im.loc[:, cond]
    print('Test feats created')
    
    pd.concat([train_im, test_im], axis=0).to_csv(os.path.join(path_to_save, 'im_feats.csv'))
    
    
    print('All saved')
    
if __name__ == '__main__':
    main()