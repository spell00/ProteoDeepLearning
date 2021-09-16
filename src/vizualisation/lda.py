#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import pandas as pd
import numpy as np

from src.utils.utils import split_train_test
from src.models.sklearn.ordination import lda
from src.utils.dataset import ms_data

# TODO validate, display valid data differently to see if they look distant
def LDA(get_data_function, data_file):
    data, labels, samples = get_data_function(data_file)
    data[np.isnan(data)] = 0
    lda(data, labels, samples, 'all')
    if 'Blk' in labels:
        for i, label in enumerate(labels):
            if label != 'Blk':
                labels[i] = 'Not Blk'

    categories = pd.Categorical(labels).codes

    lda(data, categories, samples, 'binary')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--intensities_csv", type=str, default="data/output.csv",
                        help="Path to intensities csv file")
    args = parser.parse_args()

    LDA(ms_data, args.intensities_csv)
