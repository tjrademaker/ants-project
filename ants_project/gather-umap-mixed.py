import pickle
import numpy as np
import pandas as pd

import umap
from sklearn import manifold, decomposition

df = pd.read_csv('../data/all_individuals_daysbytask.csv')
df = df.pivot_table(index=['colony','individual','setup type'],columns='behaviour')
df = df.swaplevel(0,1,axis=1)
df = df.stack()
df.index.names = ['Colony','Individual','Setup type','Day']
df.iloc[np.where(df == 0.5)] = 1
df.index = df.index.set_levels(df.index.levels[-1].astype(int), level=-1)
df.sort_index(level=-2,inplace=True) 

# Randomize data
df = df.sample(len(df),random_state=99)

print(df.head())

# UMAP
dist = 0.3
neighbors = 25
reducer = umap.UMAP(min_dist = dist, n_neighbors = neighbors, n_components = 2)
reducer.fit(df[df.index.get_level_values('Setup type') == 2])
df_umap = pd.concat((pd.DataFrame(reducer.transform(df), index = df.index),df),axis=1)

with open('../output/umap-mixed-%.2f-%d.pkl'%(dist,neighbors), 'wb') as f:
	pickle.dump(df_umap,f)

# TSNE
# perp = 20
# tsne = manifold.TSNE(n_components=2, perplexity = perp, init='pca', random_state=0)
# df_tsne = pd.concat((pd.DataFrame(tsne.fit_transform(df), index = df.index),df),axis=1)

# with open('../output/tsne-%d.pkl'%(perp), 'wb') as f:
# 	pickle.dump(df_tsne,f)

# # PCA
# pca = decomposition.PCA(n_components=2)
# df_tsne = pd.concat((pd.DataFrame(pca.fit_transform(df), index = df.index),df),axis=1)

# with open('../output/pca.pkl', 'wb') as f:
# 	pickle.dump(df_tsne,f)
