import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse

import h5py
from joblib import Parallel, delayed
import pickle

def load_subject(subject_filename):

    subject_data = None
    with h5py.File(subject_filename, 'r') as f:
        subject_data = f['SM_feature'][()]
    # It's necessary to reorient the axes, since h5py flips axis order
    subject_data = np.moveaxis(subject_data, [0,1,2,3], [3,2,1,0])
    return subject_data


def create_numpy(i, data_path, path_to_save, set_train):
    '''
    save 3d fMRI data in pickle in path_to_save
    '''
    if i in set_train:
        dir_path = 'fMRI_train'
    else:
        dir_path = 'fMRI_test'
    _p = os.path.join(data_path, dir_path)
    
    subject_filename = os.path.join(_p, str(i) + '.mat')
    subject_data = load_subject(subject_filename)
    x = subject_data.transpose((3, 2, 1, 0)).astype(np.float32)
    
    with open(os.path.join(path_to_save, str(i) + '.npy'), 'wb') as f:
        pickle.dump(x, f)
        
    return None

def main():  
    parser = argparse.ArgumentParser(description='resave 3d fMRI data')

    parser.add_argument('--data-path', default='./data/raw',
                        help='path to original images with fMRI_train and fMRI_test folders, default ./data/raw')
    parser.add_argument('--path-to-save', default='./data/imgs',
                        help='path to save new images, default ./data/imgs')
    parser.add_argument('--n-jobs', type=int, default=20,
                        help='number of jobs in multiprocessing. Default 20')
    
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

    set_train = set(train_index)
    print('Data loaded')
    
    # save data
    _ = Parallel(n_jobs=n_jobs)(delayed(create_numpy)(i, data_path, path_to_save, set_train) for i in tqdm(train_index))
    _ = Parallel(n_jobs=n_jobs)(delayed(create_numpy)(i, data_path, path_to_save, set_train) for i in tqdm(test_index))
    print('All saved')
    
if __name__ == '__main__':
    main()