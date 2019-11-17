import pickle
import umap

import numpy as np
import pandas as pd

df = pd.read_csv('../data/all_individuals_daysbytask.csv')
df = df.pivot_table(index=['colony','individual','setup type'],columns='behaviour')
df = df.swaplevel(0,1,axis=1)
df = df.stack()
df.iloc[np.where(df == 0.5)] = 1
df.index = df.index.set_levels(df.index.levels[-1].astype(int), level=-1)
df.sort_index(level=-1,inplace=True) # Test for initialization by randomizing data

df_s = df[0:0].copy()
df_s.index.names = ['Colony','Individual','Setup type','Window']
w_size = 4

for idx in np.arange(int(len(df)/20)):
    for jdx in np.arange(21-w_size):
        row = idx*20+jdx
        lvl_vals = tuple(list(df.iloc[row,:].name[0:3])+[jdx])
        df_s.loc[lvl_vals,:] = df[row:row+w_size].sum()/w_size


for dist in np.arange(0.05,0.35,0.05):
    for neighbors in np.arange(5,35,5):
        
        print(dist,neighbors)

        reducer = umap.UMAP(min_dist = dist, n_neighbors = neighbors, n_components = 2)
        df_umap = pd.concat((pd.DataFrame(reducer.fit_transform(df_s), index = df_s.index),df_s),axis=1)

        with open('../output/umap/window-%d/%.2f-%d.pkl'%(w_size,dist,neighbors), 'wb') as f:
            pickle.dump(df_umap,f)