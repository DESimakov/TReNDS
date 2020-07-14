import argparse

import pandas as pd
import numpy as np
import os
   
def main():  
    parser = argparse.ArgumentParser(description='create submission from predictions')

    parser.add_argument('--data-path', default='./data/raw',
                        help='path to original data, default ./data/raw')
    parser.add_argument('--path-site', default='./predicts/classifier',
                        help='path to detected site2 id, default ./predicts/classifier')
    parser.add_argument('--path-to-prediction', default='./predicts',
                        help='path to targets predictions, default ./predicts')
    parser.add_argument('--path-to-save', default='./predicts/submission',
                        help='path to save features, default ./data/features')

    
    args = parser.parse_args()
    data_path = args.data_path
    path_site = args.path_site
    path_to_save = args.path_to_save
    path_to_prediction = args.path_to_prediction
    
    for _path in [path_to_save]:
        if not os.path.exists(_path):
            os.makedirs(_path)
      
    extra_site = pd.DataFrame(index=np.load(os.path.join(path_site, 'site2_test_new_9735.npy')))
    ids_df = pd.read_csv(os.path.join(data_path, 'reveal_ID_site2.csv'), index_col = ['Id'])
    ids_df = ids_df.append(extra_site)
    print(f'Number of unique site2: {ids_df.index.nunique()}')
    
    # save submission without postprocessing
    sample_submission = pd.read_csv(os.path.join(data_path, 'sample_submission.csv')).sort_values('Id')
    sample_submission.loc[:, 'Predicted'] = 0
    for target in ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']:
        preds = []
        path_target = os.path.join(path_to_prediction, target)
        path_target = os.path.join(path_target, 'stack')
        for n, sub in enumerate([i for i in os.listdir(path_target) if 'stack' in i]):
            _s = pd.read_csv(os.path.join(path_target, sub))
            _s.Id = [str(i) + '_' + target for i in _s.Id]
            sample_submission.loc[sample_submission.Id.isin(_s.sort_values('Id').Id), 'Predicted'] += _s.sort_values('Id').pred.values
            preds.append(_s.sort_values('Id').pred.values)
        sample_submission.loc[sample_submission.Id.isin(_s.sort_values('Id').Id), 'Predicted'] /= (n+1)
        
    sample_submission.to_csv(os.path.join(path_to_save, "blend_aclf_x7_no_postproc.csv"), index=False)
    
    # save submission with postprocessing
    cond_age = [str(i) + '_age' for i in ids_df.index]
    cond_d11 = [str(i) + '_domain1_var1' for i in ids_df.index]
    cond_d12 = [str(i) + '_domain1_var2' for i in ids_df.index]
    cond_d21 = [str(i) + '_domain2_var1' for i in ids_df.index]
    cond_d22 = [str(i) + '_domain2_var2' for i in ids_df.index]

    epsilon_age = -1.16
    epsilon_d11 = -0.21
    epsilon_d12 = 0
    epsilon_d21 = -0.22
    epsilon_d22 = -0.2
    
    '''
    #to find this values one should manually minimize ks stat:
    from scipy.stats import ks_2samp
    _s = pd.read_csv(os.path.join(path_target, sub))
    data = pd.read_csv(os.path.join(path_to_save, "blend_aclf_x7_no_postproc.csv")).loc[sample_submission.Id.isin([str(i) + '_domain1_var1' for i in _s.Id]), :]
    cond_d11 = [str(i) + '_domain1_var1' for i in ids_df.index]
    A = data.loc[data.Id.isin(cond_d11)].Predicted
    B = data.loc[~data.Id.isin(cond_d11)].Predicted
    print(ks_2samp(A, B))
    epsilon_d11 = -0.21
    print(ks_2samp(A-epsilon_d11, B))
    '''
    data = pd.read_csv(os.path.join(path_to_save, "blend_aclf_x7_no_postproc.csv"))
    data.loc[data.Id.isin(cond_age), 'Predicted'] -= epsilon_age
    data.loc[data.Id.isin(cond_d11), 'Predicted'] -= epsilon_d11
    data.loc[data.Id.isin(cond_d12), 'Predicted'] -= epsilon_d12
    data.loc[data.Id.isin(cond_d21), 'Predicted'] -= epsilon_d21
    data.loc[data.Id.isin(cond_d22), 'Predicted'] -= epsilon_d22
    data.to_csv(os.path.join(path_to_save, "blend_aclf_x7_postproc.csv"), index=False)
    print('All saved')

if __name__ == '__main__':
    main()