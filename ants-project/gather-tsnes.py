import pickle
from sklearn import manifold

import numpy as np
import pandas as pd

df = pd.read_csv('../data/all_individuals_daysbytask.csv')
df = df.pivot_table(index=['colony','individual','setup type'],columns='behaviour')
df = df.swaplevel(0,1,axis=1)
df = df.stack()
df.index.names = ['Colony','Individual','Setup type','Day']
df.iloc[np.where(df == 0.5)] = 1
df.index = df.index.set_levels(df.index.levels[-1].astype(int), level=-1)
df.sort_index(level=-1,inplace=True)

for perp in np.arange(5,35,5):
		
		print(perp)

		tsne = manifold.TSNE(n_components=2, perplexity = perp, init='pca', random_state=0)
		df_tsne = pd.concat((pd.DataFrame(tsne.fit_transform(df), index = df.index),df),axis=1)

		with open('../output/tsne/%d.pkl'%(perp), 'wb') as f:
			pickle.dump(df_tsne,f)