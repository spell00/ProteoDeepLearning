#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import os
from tqdm import tqdm
import torch
from tqdm import tqdm
import itertools
import numpy as np
from torch.utils.data import Dataset
import random
import pandas as pd
from torchvision.transforms import Normalize
from sklearn.neighbors import KernelDensity

random.seed(42)


def ms_data(fnames, minmax=False):
    from sklearn.preprocessing import minmax_scale
    mat_datas = None
    labelss = []
    categoriess = []
    print("\nLoading samples...")
    pbar = tqdm(fnames)
    for fname in fnames[1:3]:
        mat_data = pd.read_csv(fname)
        # columns = mat_data.columns
        labels = mat_data.index.values
        categories = [int(lab.split('-')[1]) - 1 for lab in labels]
        labels = [lab.split('-')[0] for lab in labels]
        mat_data = np.asarray(mat_data)

        mat_data = mat_data.astype("float32")
        mat_data[np.isnan(mat_data)] = 0
        if mat_datas is None:
            mat_datas = mat_data
        else:
            mat_datas = np.concatenate((mat_datas, mat_data), axis=0)
        labelss.extend(labels)
        categoriess.extend(categories)
        pbar.update(1)

    if minmax:
        # mat_datas = minmax_scale(mat_datas, axis=0, feature_range=(0, 1))
        mat_datas = get_normalized(torch.Tensor(mat_datas), np.max(mat_datas)).detach().cpu().numpy()
    print(np.unique(labelss))
    print(np.unique(categoriess))
    return mat_datas, labelss, categoriess


def resize_data_1d(data, new_size=(160,)):
    initial_size_x = data.shape[0]

    new_size_x = new_size[0]

    delta_x = initial_size_x / new_size_x

    new_data = np.zeros((new_size_x))

    for x, y, z in itertools.product(range(new_size_x)):
        new_data[x][y][z] = data[int(x * delta_x)]

    return new_data


def get_normalized(x, max_val, mean_arr=False):
    x[torch.isnan(x)] = 0
    # x = Normalize(mean=[0.07777917612344026], std=[0.09724704712629319])(x.unsqueeze(0).unsqueeze(0)).squeeze()
    x = x.float()
    x = x - x.min().abs()
    if mean_arr is not True:
        if x.max() > 0:
            x /= max_val
    else:
        x /= torch.load('stats/torch/all_max')
    return x


class MSDataset(Dataset):
    def __init__(self, data, colnames, transform=get_normalized, log=True):
        self.data = data.squeeze()
        # self.max_val = torch.max(data)
        self.transform = transform
        self.log = log

        pmz = np.argwhere(np.array(colnames) == "precursorMz")[0][0] + 1
        if pmz == 1:
            self.precursormz, self.data, colnames = self.data[:, pmz], self.data[:, pmz:], colnames[pmz:]
        else:
            self.precursormz, self.data, colnames = self.data[:, pmz], \
                                                    np.concatenate((self.data[:, :pmz], self.data[:, pmz:]), 0), \
                                                    np.concatenate((colnames[:pmz], colnames[pmz:]))

        tics = np.argwhere(np.array(colnames) == "tics")[0][0] + 1
        if tics == 1:
            self.tics, self.data, colnames = self.data[:, tics], self.data[:, tics:], colnames[tics:]
        else:
            self.tics, self.data, colnames = self.data[tics], \
                                             np.concatenate((self.data[:, :tics], self.data[:, tics:]), 0),\
                                             np.concatenate((colnames[:tics], colnames[tics:]))

        rtime = np.argwhere(np.array(colnames) == "rtime")[0][0] + 1
        if pmz == 1:
            self.rtime, self.data = self.data[:, rtime], self.data[:, rtime:]
        else:
            self.rtime, self.data, colnames = self.data[:, rtime], \
                                             np.concatenate((self.data[:, :rtime], self.data[:, rtime:]), 0),\
                                             np.concatenate((colnames[:rtime], colnames[rtime:]))
        # if means is not None and stds is not None:
        #     self.data = (self.data - means) / stds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        precursormz = self.precursormz[idx]
        tics = self.tics[idx]
        rtime = self.rtime[idx]
        if self.log:
            x = torch.log1p(x)
        if self.transform is not None:
            x = self.transform.transform(x.reshape(1, -1))
        return x, precursormz, tics, rtime  # , label


class MetaMSDataset(Dataset):
    def __init__(self, data, ord, crop_size=-1, load_all=True, device='cuda'):
        self.load_all = load_all
        self.device = device
        self.ord = ord
        self.path_inputs = data
        self.crop_size = crop_size
        samples = os.listdir(data)
        self.crop_size = crop_size
        self.colnames = np.load(f"{self.path_inputs}/../columns", allow_pickle=True)
        if load_all:
            self.samples = []
            self.samples_names = []
            # pbar = tqdm(len(self.samples))
            for sample in samples:
                self.samples += [np.load('/'.join([self.path_inputs, sample]))]
                self.samples_names += [sample]
                # pbar.update(1)
        else:
            self.samples = [np.load('/'.join([self.path_inputs, samples[0]]), allow_pickle=True)]
            self.samples_names = [samples[0]]
        self.kde = KernelDensity(bandwidth=1.0, kernel='gaussian')

        # Will need to resize when multiple samples because they are not all equal in shape
        samples = np.stack(self.samples)[:, :, 3:]
        self.samples_ord = self.ord.fit_transform(samples.reshape([samples.shape[1], -1]))
        self.kde.fit(self.samples_ord)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        label = self.samples_names[idx]
        return x, label, self.colnames.tolist()


def load_checkpoint(checkpoint_path,
                    model,
                    optimizer,
                    save,
                    predict,
                    input_shape,
                    name,
                    variant,
                    nb_classes
                    ):
    losses = {
        "train": [],
        "valid": [],
    }
    if name not in os.listdir(checkpoint_path) and not predict:
        print("Creating checkpoint...")
        if save:
            save_checkpoint(model=model,
                            optimizer=optimizer,
                            learning_rate=None,
                            epoch=0,
                            checkpoint_path=checkpoint_path,
                            losses=losses,
                            name=name,
                            best_loss=None,
                            best_accuracy=None,
                            )
    checkpoint_dict = torch.load(checkpoint_path + '/' + name, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    best_loss = checkpoint_dict['best_loss']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    try:
        losses_recon = checkpoint_dict['losses_recon']
        kl_divs = checkpoint_dict['kl_divs']
    except:
        losses_recon = None
        kl_divs = None

    losses = checkpoint_dict['losses']
    print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, epoch))
    return model, optimizer, epoch, losses, kl_divs, losses_recon, best_loss


def save_checkpoint(model,
                    optimizer,
                    learning_rate,
                    epoch,
                    checkpoint_path,
                    losses,
                    best_loss,
                    best_accuracy,
                    name="cnn",
                    ):
    # model.load_state_dict(model.state_dict())
    print(checkpoint_path)
    print(name)
    os.makedirs(f"{checkpoint_path}/{name}", exist_ok=True)
    torch.save({'model': model,
                'losses': losses,
                'best_loss': best_loss,
                'best_accuracy': best_accuracy,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, f"{checkpoint_path}/{name}")


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


class validation_spliter:
    def __init__(self, dataset, cv):
        self.cv = cv
        self.dataset = dataset
        self.current_cv = 0
        self.val_offset = int(np.floor(len(self.dataset) / self.cv))
        self.current_pos = 0

    def __next__(self):
        self.current_cv += 1
        # if self.current_cv == self.cv:
        #     val_offset = len(self.dataset) - self.current_pos
        # else:
        #     val_offset = self.val_offset
        partial_dataset = PartialDataset(self.dataset, 0, self.val_offset), \
                          PartialDataset(self.dataset, self.val_offset, len(self.dataset) - self.val_offset)

        # Move the samples currently used for the validation set at the end for the next split
        tmp = self.dataset.samples[:self.val_offset]
        self.dataset.samples = np.concatenate([self.dataset.samples[self.val_offset:], tmp], 0)

        return partial_dataset
