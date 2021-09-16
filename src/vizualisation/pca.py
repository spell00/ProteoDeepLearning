#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import os
import pandas as pd
import numpy as np

from src.utils.utils import split_train_test
from src.utils.dataset import ms_data
from src.models.sklearn.ordination import pca
import matplotlib.pyplot as plt


# TODO validate, display valid data differently to see if they look distant
def PCA(get_data_function, args):
    # labels = pd.read_csv(labels_file, header=0).loc[0]
    gs = args.g.split('-')
    sampless = None
    labelss = None
    datas = None
    for g in gs:
        data_files = os.listdir(f"{args.intensities_dir}/{g}/m2/train")
        data_files = [f"{args.intensities_dir}/{g}/m2/train/{file}" for file in data_files]
        data, labels, samples = get_data_function(data_files, minmax=True)
        data[np.isnan(data)] = 0
        if labelss is None:
            sampless = samples
            labelss = labels
            datas = data
        else:
            sampless = np.concatenate((sampless, samples), axis=0)
            labelss = np.concatenate((labelss, labels), axis=0)
            datas = np.concatenate((datas, data), axis=0)
        del data, labels, samples
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    pca(datas, labelss, ax)
    os.makedirs("results/images/", exist_ok=True)
    plt.savefig(fname=f"results/images/pca_{args.g}_labelwise.png", dpi=100)
    plt.close()

    labelss = [f"{label}-{sample}" for label, sample in zip(labelss, sampless)]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    pca(datas, labelss, ax)
    plt.savefig(fname=f"results/images/pca_{args.g}_samplewise.png", dpi=100)
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--intensities_dir", type=str, default="inputs_bin1",
                        help="Path to intensities csv file")
    parser.add_argument("--g", type=str, default="g30-g90",
                        help="Path to intensities csv file")
    args = parser.parse_args()

    PCA(ms_data, args)
