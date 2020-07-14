import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle
from sklearn.decomposition import IncrementalPCA, MiniBatchDictionaryLearning
import gc

def load_subject(subject_filename):
    with open(subject_filename, 'rb') as f:
        subject_data = pickle.load(f)
    return subject_data

class ImageLoader():
    def __init__(self, transforms=None):
        self.transforms = transforms
        pass
    
    def transform(self, X, y=None):
        X = load_subject(X)
        if self.transforms is not None:
            X = self.transforms(image=X)['image']
        
        return X  
    
def main():  
    parser = argparse.ArgumentParser(description='train pca and dl features')

    parser.add_argument('--data-path', default='./data/raw',
                        help='path to original data, default ./data/raw')
    parser.add_argument('--imgs-path', default='./data/imgs',
                        help='path to resaved images, default ./data/imgs')
    parser.add_argument('--path-to-save', default='./data/features',
                        help='path to save features, default ./data/features')
    parser.add_argument('--path-to-save-model', default='./models/pca',
                        help='path to save models, default ./models/pca')
    
    args = parser.parse_args()
    data_path = args.data_path
    imgs_path = args.imgs_path
    path_to_save = args.path_to_save
    path_to_save_model = args.path_to_save_model
    
    for _path in [path_to_save, path_to_save_model,
                  os.path.join(path_to_save, '100dl_feats'),
                  os.path.join(path_to_save, '200pca_feats')]:
        if not os.path.exists(_path):
            os.makedirs(_path)
  
    
    loading = pd.read_csv(os.path.join(data_path, 'loading.csv'), index_col = ['Id'])
    
    # creates pathes to all images
    img_path = pd.DataFrame(index=loading.index, columns=['path'])
    for index in img_path.index:
        path = str(index) + '.npy'
        img_path.loc[index, 'path'] = os.path.join(imgs_path, path)
    
    
    # start train and inference of pca feats
    print('PCA started. ~13 hours')
    
    for k in range(0, 6):
        ##fit
        pca = IncrementalPCA(n_components=200)
        batch = []
        for n, i in enumerate(tqdm(img_path.values)):
            f = ImageLoader().transform(i[0])
            f = f[k*10:(k+1)*10].flatten()
            batch.append(f)
            if (n + 1) % 200 == 0:
                batch = np.array(batch)
                pca.partial_fit(batch)
                del batch
                gc.collect()
                batch = []
        ##save pca
        _p = os.path.join(path_to_save_model, f'200pca_3d_k{k}.pickle')
        with open(_p, 'wb') as f:
            pickle.dump(pca, f)
        
        ##transform
        res = []
        batch = []
        for n, i in enumerate(tqdm(img_path.values)):
            f = ImageLoader().transform(i[0])
            f = f[k*10:(k+1)*10].flatten()
            batch.append(f)
            if (n + 1) % 200 == 0:
                batch = np.array(batch)
                res.append(pca.transform(batch))
                del batch
                gc.collect()
                batch = []
        lb = len(batch)        
        if lb > 0:
            batch = np.array(batch)
            if lb == 1:
                res.append(pca.transform(batch.reshape(1, -1)))
            else:
                res.append(pca.transform(batch))
        
        ##save df
        res = np.array(res)
        df_res = pd.DataFrame(np.vstack(res), index=loading.index, columns=[f'200PCA_k{k}_' + str(i) for i in range(200)])
        _p = os.path.join(path_to_save, f'200pca_feats/200pca_3d_k{k}.csv')
        df_res.to_csv(_p)
        
    print('Dictionary learning started. ~47 hours')
    n_k = 100
    for k in range(0, 6):
        
        ##fit
        pca = MiniBatchDictionaryLearning(n_components=n_k, random_state=0, n_iter=10, batch_size=n_k)
        batch = []
        for n, i in enumerate(tqdm(img_path.values)):
            f = ImageLoader().transform(i[0])
            f = f[k*10:(k+1)*10].flatten()
            batch.append(f)
            if (n + 1) % 100 == 0:
                batch = np.array(batch)
                pca.partial_fit(batch)
                del batch
                gc.collect()
                batch = []
        ##save pca  
        _p = os.path.join(path_to_save_model, f'dl_3d_k{k}.pickle')
        with open(_p, 'wb') as f:
            pickle.dump(pca, f)
        
        ##transform
        res = []
        batch = []
        for n, i in enumerate(tqdm(img_path.values)):
            f = ImageLoader().transform(i[0])
            f = f[k*10:(k+1)*10].flatten()
            batch.append(f)
            if (n + 1) % 100 == 0:
                batch = np.array(batch)
                res.append(pca.transform(batch))
                del batch
                gc.collect()
                batch = []
    
        lb = len(batch)        
        if lb > 0:
            batch = np.array(batch)
            if lb == 1:
                res.append(pca.transform(batch.reshape(1, -1)))
            else:
                res.append(pca.transform(batch))
        
        ##save df
        res = np.array(res)
        df_res = pd.DataFrame(np.vstack(res), index=loading.index, columns=[f'dl_k{k}_' + str(i) for i in range(n_k)])
        _p = os.path.join(path_to_save, f'100dl_feats/dl_3d_k{k}.csv')
        df_res.to_csv(_p)
        
    #resave results        
    _p = os.path.join(path_to_save, '100dl_feats/dl_3d_k0.csv')    
    data_pca = pd.read_csv(_p)
    for i in range(1, 6):
        _p = os.path.join(path_to_save, '100dl_feats/dl_3d_k{}.csv'.format(i))    

        part = pd.read_csv(_p)
        del part['Id']
        data_pca = pd.concat((data_pca, part), axis=1)
    data_pca.to_csv(os.path.join(path_to_save, 'dl_feats.csv'), index=None)
    
    print('All saved')

if __name__ == '__main__':
    main()