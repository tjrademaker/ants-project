import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.cluster import DBSCAN

import seaborn as sns


### Filter data on conditions (colony, individual, setup type, clusters) ###
def build_condition_filter(df,pairs):

    condition_filter = [True] * len(df)

    for (condition,value) in pairs:

        if isinstance(value, tuple):
            temp_filter = [False] * len(df)
            for val in value:
                temp_filter = (temp_filter) | (df.index.get_level_values(condition) == val)
            condition_filter = (condition_filter) & (temp_filter)

        else:
            condition_filter = (condition_filter) & (df.index.get_level_values(condition) == value)

    return condition_filter


### Filter to exclude datapoints that are outside of the chosen range ###
def build_position_filter(df,lims):

    xlims,ylims = lims
    x = (df[0] >= xlims[0]) & (df[0] <= xlims[1])
    y = (df[1] >= ylims[0]) & (df[1] <= ylims[1])

    return (x & y)


### Determine clusters, return labels of clusters ###
def determine_clusters(x_f,y_f,epsilon,min_samples,filename):

    df = pickle.load(open('../output/%s'%filename,'rb'))
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(df[[0,1]].values)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    fig,ax=plt.subplots(figsize=(10,10))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for points outside of cluster.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = df[[0,1]].values[class_member_mask & core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            markeredgecolor='k', markersize=14)

        xy = df[[0,1]].values[class_member_mask & ~core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            markeredgecolor='k', markersize=6)
    plt.title('Estimated number of clusters: %d; eps = %.2f; min_samples = %d' % (n_clusters,epsilon,min_samples))

    # Set x_range and y_range in terms of min and max of embedded coordinates
    x_range,y_range = (df[0].max() - df[0].min(),df[1].max() - df[1].min())
    lims = [(df[0].min()+x_f[0]*x_range,df[0].min()+x_f[1]*x_range), (df[1].min()+y_f[0]*y_range,df[1].min()+y_f[1]*y_range)]

    set_axis_umap(ax,lims,(x_f,y_f))

    return db.labels_;


### Plot the histograms ###
def plot_hist(df,df_c):

    # 2 x 7 = 14 plots for 13 behaviours
    fig,axes = plt.subplots(2,7,figsize=(20,3))

    # Loop over the behaviours, and make the days columns. Plot as a bar chart (which is essentially a histogram)
    for i,(ax,var) in enumerate(zip(axes.flatten(),df.columns[2:])):
        all_data = df[var].unstack('Day').sum()/len(df)
        select_data = df_c[var].unstack('Day').sum()/len(df_c)
        all_data.plot(kind = 'bar',ax = ax, color = 'blue', width=1, alpha = 0.5, align = 'center')
        select_data.plot(kind = 'bar', ax = ax, color = 'red', width=1, alpha = 0.5, align = 'center')

        # Set axes labels. Need to think about y axes labels and scaling
        ax.set_xticks([0,5,10,15,20])
        ax.set_title(var,y=0.7,fontsize=16)
        ax.set_xlim([0,22])
        if i < 7: ax.set_xlabel(''); ax.set_xticklabels([])
        else: ax.set_xlabel('Time (days)'); ax.set_xticklabels([0,5,10,15,20],rotation=0)

    # Create a legend
    blue_patch = mpatches.Patch(color='blue', alpha = 0.5, label='All')
    red_patch = mpatches.Patch(color='red', alpha = 0.5, label='Select')
    ax.legend(handles=[blue_patch, red_patch],markerscale=20,bbox_to_anchor=(1.7,1))

    axes[1,6].axis('off');


### Set axis labels for UMAP plot
def set_axis_umap(ax,lims,lims_f):

    ax.set_xlabel('umap1 (rescaled coordinates)',fontsize=16)
    ax.set_ylabel('umap2 (rescaled coordinates)',fontsize=16)

    # Lims contains the desired range in UMAP coordinates
    [(xmin,xmax),(ymin,ymax)] = lims
    ax.set_xlim([xmin-(xmax-xmin)/100,xmax+(xmax-xmin)/100])
    ax.set_ylim([ymin-(ymax-ymin)/100,ymax+(ymax-ymin)/100])

    ax.set_xticks(np.linspace(xmin,xmax,11))
    ax.set_yticks(np.linspace(ymin,ymax,11))

    # Lims_f contains the desired range in fractions from 0 to 1
    [(xmin_f,xmax_f),(ymin_f,ymax_f)] = lims_f
    ax.set_xticklabels(['%.2f'%i for i in np.linspace(xmin_f,xmax_f,11)])
    ax.set_yticklabels(['%.2f'%j for j in np.linspace(ymin_f,ymax_f,11)])


### Plot UMAP projection in a given range (lims). Points that are included in the condition (df) are colored, points that are excluded (df_gray) are gray
def plot_umap(df,df_gray,lims,lims_f):

    fig,ax = plt.subplots(figsize=(10,10))

    # Plot the data
    h = ax.scatter(df[0],df[1], s = np.min([1e3/len(df),10]), c = np.array(df.index.get_level_values('Day').astype('float')), vmin = 2, vmax = 22)
    ax.scatter(df_gray[0],df_gray[1], s = 1e2/(len(df)+len(df_gray)), c = 'gray')

    # Plot the colorbar; could change this to cluster
    cbar = plt.colorbar(h,orientation='horizontal')
    cbar.set_ticks([5,10,15,20])
    cbar.ax.set_yticklabels([5,10,15,20])
    cbar.ax.set_title('Time (days)')

    set_axis_umap(ax,lims,lims_f)


### Wrapper for interactive function ###
def interact_box(labels,x_f,y_f,colony,individual=0,setup_type=0,cluster=0):

    df = pickle.load(open('../data/umap-cluster.pkl','rb'))

    # Set x_range and y_range in terms of min and max of embedded coordinates
    x_range,y_range = (df[0].max() - df[1].min(),df[1].max() - df[1].min())
    print(x_range,y_range)
    lims = [(df[0].min()+x_f[0]*x_range,df[0].min()+x_f[1]*x_range), (df[1].min()+y_f[0]*y_range,df[1].min()+y_f[1]*y_range)]

    # Join the full condition in one list called pairs. The condition filter will take pairs and output a list of datapoints to include and to exclude
    pairs = []

    colonies = np.unique(df.index.get_level_values('Colony'))
    individuals = np.unique(df.index.get_level_values('Individual'))
    clusters = np.unique(labels)

    if 'all' not in colony:
        pairs.append(('Colony',colony))
    if individual > 0:
        i = individuals[individual-1]
        pairs.append(('Individual',i))
        print("Individual = %s"%i)
    if setup_type > 0:
        s = [1,2][setup_type-1]
        pairs.append(('Setup type',s))
        print("Setup type = %s"%s)
    if cluster > 0:
        cl = clusters[cluster-1]
        pairs.append(('Cluster',cl))
        print("Cluster = %s"%cl)

    # Build position filter and condition filter from the data that pass the position filter (df_p)
    position = build_position_filter(df,lims)
    df_p = df[position]

    condition = build_condition_filter(df_p,pairs)

    if np.sum(condition) == 0:
        print('NO DATA')
    else:
        print('Fraction of total points = %.3f'%(np.sum(condition)/len(df)))
        
        df_pc = df_p[condition] # Colored datapoints in the condition
        df_pnc = df_p[[not cond for cond in condition]] # Gray datapoints not in the condition

        plot_umap(df_pc,df_pnc,lims,[x_f,y_f])
        plot_hist(df_p,df_pc)

### Wrapper for interactive function ###
def save_figure(labels,x_f,y_f,colony,individual,setup_type,cluster):

    df = pickle.load(open('../data/umap-cluster-shuffle.pkl','rb'))

    # Set x_range and y_range in terms of min and max of embedded coordinates
    x_range,y_range = (df[0].max() - df[1].min(),df[1].max() - df[1].min())
    lims = [(df[0].min()+x_f[0]*x_range,df[0].min()+x_f[1]*x_range), (df[1].min()+y_f[0]*y_range,df[1].min()+y_f[1]*y_range)]

    # Join the full condition in one list called pairs. The condition filter will take pairs and output a list of datapoints to include and to exclude
    pairs = []

    colonies = np.unique(df.index.get_level_values('Colony'))
    individuals = np.unique(df.index.get_level_values('Individual'))
    clusters = np.unique(labels)

    if 'all' not in colony:
        pairs.append(('Colony',colony))
    if individual > 0:
        i = individuals[individual-1]
        pairs.append(('Individual',i))
        print("Individual = %s"%i)
    if setup_type > 0:
        s = [1,2][setup_type-1]
        pairs.append(('Setup type',s))
        print("Setup type = %s"%s)
    if cluster > 0:
        cl = clusters[cluster-1]
        pairs.append(('Cluster',cl))
        print("Cluster = %s"%cl)

    # Build position filter and condition filter from the data that pass the position filter (df_p)
    position = build_position_filter(df,lims)
    df_p = df[position]

    condition = build_condition_filter(df_p,pairs)

    if np.sum(condition) == 0:
        print('NO DATA')
    else:
        print('Fraction of total points = %.3f'%(np.sum(condition)/len(df)))
        
        df_pc = df_p[condition] # Colored datapoints in the condition
        df_pnc = df_p[[not cond for cond in condition]] # Gray datapoints not in the condition

        plot_umap(df_pc,df_pnc,lims,[x_f,y_f])
        plt.savefig('../figures/umap/%s_%d_%d_%d.pdf'%(colony,individual,setup_type,cluster))
        plot_hist(df_p,df_pc)
        plt.savefig('../figures/hist/%s_%d_%d_%d.pdf'%(colony,individual,setup_type,cluster))
