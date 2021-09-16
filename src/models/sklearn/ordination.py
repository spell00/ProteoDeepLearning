#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
from src.utils.utils import split_train_test, getScalerFromString


def rand_jitter(arr):
    return arr + np.random.randn(len(arr)) * 0.01


def pca(data, labels, ax):
    data[np.isnan(data)] = 0
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data, labels)

    principal_df = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
    labels_df = pd.DataFrame(labels)
    labels_df.index = principal_df.index
    final_df = pd.DataFrame(np.concatenate((principal_df.values, labels_df.values), axis=1),
                            columns=list(principal_df.columns) + ['label'])

    ev = pca.explained_variance_ratio_
    pc1 = 'Component_1 : ' + str(np.round(ev[0] * 100, decimals=2)) + "%"
    pc2 = 'Component_1 : ' + str(np.round(ev[1] * 100, decimals=2)) + "%"
    ax.set_xlabel(pc1, fontsize=15)
    ax.set_ylabel(pc2, fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    cmap = plt.cm.coolwarm

    num_targets = len(set(labels))

    cols = cmap(np.linspace(0, 1, num_targets))
    colors = rcParams['axes.prop_cycle'] = cycler(color=cols)
    # colors = ["g", "b", "k", "r"]
    colors_list = []
    data1_list = []
    data2_list = []
    for t, target in enumerate(list(set(labels))):
        indices_to_keep = [True if x == target else False for x in
                           list(final_df.label)]  # 0 is the name of the column with target values
        data1 = list(final_df.loc[indices_to_keep, 'principal component 1'])
        data2 = list(final_df.loc[indices_to_keep, 'principal component 2'])
        try:
            assert np.sum(np.isnan(data1)) == 0 and np.sum(np.isnan(data2)) == 0
        except:
            print("Nans were detected. Please verify the DataFrame...")
            exit()
        data1_list += [data1]
        data2_list += [data2]
        colors_list += [np.array([[cols[t]] * len(data1)])]

    colors_vector = np.hstack(colors_list).squeeze()
    data1_vector = np.hstack(data1_list).reshape(-1, 1)
    data2_vector = np.hstack(data2_list).reshape(-1, 1)
    data_colors_vector = np.concatenate((data1_vector, data2_vector, colors_vector), axis=1)
    data1 = data_colors_vector[:, 0]
    data2 = data_colors_vector[:, 1]
    col = data_colors_vector[:, 2:]

    ax.scatter(data1, data2, s=10, alpha=0.5, c=col, label=labels)
    custom_lines = [Line2D([0], [0], color=cmap(x), lw=4) for x in np.linspace(0, 1, num_targets)]

    ax.legend(custom_lines, list(set(labels)))

    principal_components = pca.transform(data)

    principal_df = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
    labels_df = pd.DataFrame(labels)
    labels_df.index = principal_df.index
    final_df = pd.DataFrame(np.concatenate((principal_df.values, labels_df.values), axis=1),
                            columns=list(principal_df.columns) + ['label'])

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111)
    ev = pca.explained_variance_ratio_
    pc1 = 'Component_1 : ' + str(np.round(ev[0] * 100, decimals=2)) + "%"
    pc2 = 'Component_1 : ' + str(np.round(ev[1] * 100, decimals=2)) + "%"
    ax.set_xlabel(pc1, fontsize=15)
    ax.set_ylabel(pc2, fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    return final_df

def get_results(data, labels, model, name, ax):
    n_labels = len(set(labels))
    if n_labels > 2:
        n_comps = 2
    else:
        n_comps = 1
    ord = model(n_components=n_comps)
    principal_components = ord.fit_transform(data, labels)

    try:
        principal_df = pd.DataFrame(data=principal_components,
                                    columns=['principal component 1', 'principal component 2'])
    except:
        principal_df = pd.DataFrame(data=principal_components, columns=['principal component 1'])

    labels_df = pd.DataFrame(labels)
    labels_df.index = principal_df.index
    final_df = pd.DataFrame(np.concatenate((principal_df.values, labels_df.values), axis=1),
                            columns=list(principal_df.columns) + ['label'])

    ev = ord.explained_variance_ratio_
    pc1 = 'Component_1 : ' + str(np.round(ev[0] * 100, decimals=2)) + "%"
    ax.set_xlabel(pc1, fontsize=15)
    try:
        pc2 = 'Component_2 : ' + str(np.round(ev[1] * 100, decimals=2)) + "%"
        ax.set_ylabel(pc2, fontsize=15)
    except:
        pass
    ax.set_title(f'2 component {name}', fontsize=20)
    cmap = plt.cm.coolwarm

    num_targets = len(set(labels))

    cols = cmap(np.linspace(0, 1, num_targets))
    colors = rcParams['axes.prop_cycle'] = cycler(color=cols)
    colors2 = ["g", "b", "k", "r"]
    colors_list = []
    data1_list = []
    data2_list = []
    for t, target in enumerate(reversed(list(set(labels)))):
        indices_to_keep = [True if x == target else False for x in
                           list(final_df.label)]  # 0 is the name of the column with target values
        data1 = list(final_df.loc[indices_to_keep, 'principal component 1'])
        try:
            data2 = list(final_df.loc[indices_to_keep, 'principal component 2'])
        except:
            data2 = False
        try:
            assert np.sum(np.isnan(data1)) == 0 and np.sum(np.isnan(data2)) == 0
        except:
            print("Nans were detected. Please verify the DataFrame...")
            exit()
        data1_list += [data1]
        if data2:
            data2_list += [data2]
            colors_list += [np.array([[cols[t]] * len(data1)])]
        else:
            colors_list += [np.array([[colors2[t]] * len(data1)])]

    data1_vector = np.hstack(data1_list).reshape(-1, 1)
    colors_vector = np.hstack(colors_list).squeeze()
    if len(data2_list) > 0:
        data2_vector = np.hstack(data2_list).reshape(-1, 1)
        data_colors_vector = np.concatenate((data1_vector, data2_vector, colors_vector), axis=1)
        data2 = data_colors_vector[:, 1]
        col = data_colors_vector[:, 2:]
        data1 = data_colors_vector[:, 0]
    else:
        col = np.array(colors_vector)
        data1 = data1_vector

    if len(data2_list) > 0:
        ax.scatter(data1, data2, s=10, alpha=0.5, c=col, label=labels)
        custom_lines = [Line2D([0], [0], color=cmap(x), lw=4) for x in np.linspace(0, 1, num_targets)]
        ax.legend(custom_lines, list(set(labels)))
    else:
        data1 = data1.reshape(-1)
        zeros = np.zeros_like(data1)
        ax.scatter(data1.reshape(-1), rand_jitter(zeros), s=10, alpha=0.5, c=col, label=labels)


def lda(data, labels_str, categories, name, ord_name):
    data[np.isnan(data)] = 0
    unique_labels = np.unique(labels_str)
    labels = np.array([np.argwhere(lab == unique_labels)[0][0] for lab in labels_str])

    train_inds, test_inds, train_cats, train_samples = split_train_test(labels, categories)
    all_train_indices = [s for s, lab in enumerate(categories) if lab in train_inds]
    test_indices = [s for s, lab in enumerate(categories) if lab in test_inds]

    data_train = data[all_train_indices]
    data_test = data[test_indices]
    labels_train = labels[all_train_indices]
    labels_test = labels[test_indices]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    get_results(data_train, labels_train, LDA, ord_name, ax)
    get_results(data_test, labels_test, LDA, ord_name, ax)
    # data = data_train
    # labels = labels_train

    os.makedirs("results/images/", exist_ok=True)
    plt.savefig(fname=f"results/images/lda_{name}.png", dpi=100)
    plt.close()
