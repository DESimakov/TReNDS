import os
import pickle
import itertools
from copy import deepcopy
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from scipy.optimize import differential_evolution
from tqdm.notebook import tqdm

def NMAE(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0) / np.sum(y_true, axis=0))

class TrendsModelSklearn():
    def __init__(self, zoo: list, seed=0):
        self.zoo = zoo
        self.seed = seed
    
    def prepare_label(self):
        # create mask and label
        self.mask = self.label.notnull()
        self.label = self.label.dropna()
    
    def gen_folds(self):
        # drop nans
        self.prepare_label()
        self.folds = list(KFold(n_splits=10, shuffle=True, random_state=self.seed).split(self.label.values))
            
    def get_data(self, data: pd.DataFrame, is_test=False):
        if is_test:
            return data.values
        else:
            return data.loc[self.mask, :].values, self.label.values
        
    def fit_model(self, model, xtrain, ytrain):
        
        scores = []
        models = []
        oof_pred = np.zeros(xtrain.shape[0])
        
        for n, (tr, te) in enumerate(self.folds):
            xtr, xte = xtrain[tr], xtrain[te]
            ytr, yte = ytrain[tr], ytrain[te]
            
            model_local = deepcopy(model)
            model_local.fit(xtr, ytr)
            pred = model_local.predict(xte)
            
            # save results
            models.append(model_local)
            scores.append(NMAE(yte, pred))
            oof_pred[te] = pred
            
            print('Fold {} NMAE: {:.5f}'.format(n, scores[-1]))
        
        print('\nNMAE mean: {:.5f}'.format(np.mean(scores)))
        print('NMAE OOF: {:.5f}\n'.format(NMAE(ytrain, oof_pred)))
        return models, oof_pred
        
    def fit(self, datasets: list, label: pd.Series):
        
        self.label = label
        self.models_trained = []
        self.oof_preds = []
        self.oof_scores = []
        
        # prepare folds
        self.gen_folds()
        
        for dataset, model in zip(datasets, self.zoo):
            # get data
            xtrain, ytrain = self.get_data(dataset)                
            
            # train models
            models, oof_pred = self.fit_model(model, xtrain, ytrain)
            self.models_trained.append(models)
            self.oof_preds.append(oof_pred)
            self.oof_scores.append(np.round(NMAE(ytrain, oof_pred), 5))
        
    def save_oofs(self, names, folder):
        for n, name in enumerate(names):
            with open(folder + '/{}_oof.pickle'.format(name), 'wb') as handle:
                pickle.dump(self.oof_preds[n], handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_models(self, names, folder): 
        for n, name in enumerate(names):
            with open(folder + '/{}_model.pickle'.format(name), 'wb') as handle:
                pickle.dump(self.models_trained[n], handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    def load_models(self, file):
        # put cycle here
        # load models
        with open(file, 'rb') as handle:
            self.models_loaded = pickle.load(handle)
    
    def predict(self, datasets, model_names, is_blend=True):
        self.preds = {}
        for dataset, model_pool, name in zip(datasets, self.models_trained, model_names):
            # get data
            xtest = self.get_data(dataset, is_test=True)
            
            pool_preds = []
            for model in model_pool:
                pool_preds.append(model.predict(xtest))
            
            self.preds[name] = pool_preds

        self.test_pred = [np.mean(self.preds[name], axis=0) for name in model_names]
        if is_blend:
            return np.dot(self.weights, self.test_pred)
        else:
            return self.test_pred
    
    def blend_oof(self):

        def compute_blend(w):
            return NMAE(self.label, np.dot(w, np.array(self.oof_preds)))
        
        print('Correlation matrix')
        print(np.corrcoef(self.oof_preds), '\n')
        
        bounds = [(0, 1)]*len(self.oof_preds)
        res = differential_evolution(compute_blend, bounds, seed=0)
        ws = res.x
        self.blend = np.dot(ws, np.array(self.oof_preds))
        self.weights = ws

        print('Selected weights: ', ws)
        print('Blend NMAE: {:.5f}'.format(res.fun))
        return self.blend
