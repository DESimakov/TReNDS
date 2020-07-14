import itertools
import numpy as np
import pandas as pd

def parse_fnc(x):
    return [c[:-4] for c in x.split('_vs_')]

def compute_fnc_agg(df, pair_cols):
    df_res = pd.DataFrame()
    
    for c in pair_cols.keys():
        df_res[c[0] + '_' + c[1] + '_mean'] = df[pair_cols[c]].mean(axis=1)
        df_res[c[0] + '_' + c[1] + '_min'] = df[pair_cols[c]].min(axis=1)
        df_res[c[0] + '_' + c[1] + '_max'] = df[pair_cols[c]].max(axis=1)
        df_res[c[0] + '_' + c[1] + '_std'] = df[pair_cols[c]].std(axis=1).fillna(0)

    return df_res

# load data
print('Loading data...')
df = pd.read_csv('./data/raw/fnc.csv')
fnc_cols = sorted(df.columns[1:])

# generate pair of cols
pair_cols = {}
colz = ['SCN', 'ADN', 'SMN', 'VSN', 'CON', 'DMN', 'CBN']
for pair in itertools.combinations_with_replacement(colz, 2):
    pair_cols[pair] = [c for c in fnc_cols if set(parse_fnc(c)) == set(pair)]

# compute features
print('Computing features...')
agg_feats = compute_fnc_agg(df, pair_cols)
agg_feats = agg_feats[agg_feats.std()[agg_feats.std() != 0].index]
agg_feats.columns = ['AGG_' + c.lower() for c in agg_feats.columns]
#agg_feats['Id'] = df['Id']
agg_feats = pd.concat((df[['Id']], agg_feats), axis=1)
agg_feats.to_csv('./data/features/agg_feats.csv', index=False)