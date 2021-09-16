#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import pandas as pd
import numpy as np
import torch
import torch


def position_encoding(seq_len: int, dim_model: int, device: torch.device = torch.device("cpu")):
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / 1e4 ** (dim // dim_model)

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


def get_labels(fname):
    meta = pd.read_excel(fname, header=0)
    toremove = pd.isnull(meta.values[:, 0])
    tokeep = [i for i, x in enumerate(toremove) if x == 0]

    meta = meta.iloc[tokeep, :]
    samples_classes = meta['Pathological type']
    classes = np.unique(samples_classes)

    return classes, samples_classes


def to_categorical(y, num_classes, dtype=torch.int):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes, dtype=dtype)[y]


def get_samples_names(labels):
    samples = {s: [] for s in np.unique(labels['label'])}

    new_keys = []
    categories = []
    nums = []
    for i, label in enumerate(samples.keys()):
        tmp = label.split('-')
        lab = tmp[0].split('c..')[1]
        num = tmp[1]
        cat = 0
        if lab != 'Normal':
            cat = 1
            lab = 'Not Normal'
        new_keys += [f'{lab}-{num}']
        categories += [cat]
        if num not in nums:
            nums += [int(num)]
    # samples = dict(zip(new_keys, list(samples.values())))

    return categories, nums


def split_labels_indices(labels, train_inds):
    train_indices = []
    test_indices = []
    for j, sample in enumerate(list(labels)):
        if sample in train_inds:
            train_indices += [j]
        else:
            test_indices += [j]

    assert len(test_indices) != 0
    assert len(train_indices) != 0

    return train_indices, test_indices


def split_train_test(labels, categories):
    from sklearn.model_selection import StratifiedKFold
    # First, get all unique samples and their category
    unique_samples = []
    unique_cats = []
    for sample, cat in zip(labels, categories):
        if cat not in unique_cats:
            unique_samples += [sample]
            unique_cats += [cat]
    # StratifiedKFold with n_splits of 5 to ranmdomly split 80/20.
    # Used only once for train/test split.
    # The train split needs to be split again into train/valid sets later

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    train_inds, test_inds = next(skf.split(unique_cats, unique_samples))

    # After the samples are split, we get the duplicates of all samples.
    train_samples, test_samples = [unique_samples[s] for s in train_inds], [unique_samples[s] for s in test_inds]
    train_cats = [unique_cats[ind] for ind in train_inds]

    assert len(unique_samples) == len(train_inds) + len(test_inds)
    assert len([x for x in test_inds if x in train_inds]) == 0
    # assert len([x for x in test_samples if x in train_samples]) == 0
    # assert len(np.unique([unique_cats[ind] for ind in test_samples])) > 1

    return train_inds, test_inds, train_cats, train_samples


def getScalerFromString(scaler_str):
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer, StandardScaler
    if str(scaler_str) == 'normalizer':
        scaler = Normalizer
    elif str(scaler_str) == 'standard':
        scaler = StandardScaler
    elif str(scaler_str) == 'minmax':
        scaler = MinMaxScaler
    elif str(scaler_str) == "robust":
        scaler = RobustScaler
    else:
        exit(f"Invalid scaler {scaler_str}")
    return scaler


def keras_confusion_matrix(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    return tp, tn, fp, fn


class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.parent_ds[i + self.offset]


def validation_split(dataset, val_share=0.1):
    """
       Split a (training and vaidation combined) dataset into training and validation.
       Note that to be statistically sound, the items in the dataset should be statistically
       independent (e.g. not sorted by class, not several instances of the same dataset that
       could end up in either set).
       inputs_bin01:
          dataset:   ("training") dataset to split into training and validation
          val_share: fraction of validation data (should be 0<val_share<1, default: 0.1)
       returns: input dataset split into test_ds, val_ds
       """
    val_offset = int(len(dataset) * (1 - val_share))
    return PartialDataset(dataset, 0, val_offset), PartialDataset(dataset, val_offset, len(dataset) - val_offset)


class validation_spliter:
    def __init__(self, dataset, cv):
        self.cv = cv
        self.dataset = dataset
        self.current_cv = 0
        self.val_offset = int(np.floor(len(dataset) / self.cv))
        self.current_pos = 0

    def __next__(self):
        self.current_cv += 1
        # if self.current_cv == self.cv:
        #     val_offset = len(self.dataset) - self.current_pos
        # else:
        #     val_offset = self.val_offset
        partial_dataset = PartialDataset(self.dataset, 0, self.val_offset), PartialDataset(self.dataset,
                                                                                           self.val_offset,
                                                                                           len(self.dataset) - self.val_offset)

        # Move the samples currently used for the validation set at the end for the next split
        tmp = self.dataset.samples[:self.val_offset]
        self.dataset.samples = np.concatenate([self.dataset.samples[self.val_offset:], tmp])

        return partial_dataset
