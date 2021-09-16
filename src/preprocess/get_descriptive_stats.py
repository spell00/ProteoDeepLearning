import os
import torch
import random
import warnings
import argparse
import numpy as np
from torch.utils.data import DataLoader
from src.utils.dataset import MSDataset, MetaMSDataset
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')

device = 'cuda'


def train(train_set, label, is_log, is_norm):
    train_loader = DataLoader(train_set,
                              num_workers=0,
                              shuffle=True,
                              batch_size=8,
                              pin_memory=True,
                              drop_last=True)
    all_ms(train_loader, label, is_log, is_norm)
    mean = mean_ms(train_loader, label, is_log, is_norm)
    std_ms(train_loader, label, mean, is_log, is_norm)
    return mean


def all_ms(train_loader, label, is_log, is_norm):
    plt.figure(figsize=(20, 20))
    for j, ms in enumerate(train_loader):
        for m in ms:
            plt.plot(m)
        if j % 100:
            break
    plt.title(f"All ms")
    plt.xlabel('m/z')
    plt.ylabel('Normalized intensity')
    os.makedirs(f"stats/{label[0]}", exist_ok=True)
    plt.savefig(f"stats/{label[0]}/all_log{is_log}_norm{is_norm}_ms")
    plt.close()


def mean_ms(train_loader, label, is_log, is_norm):
    mean = None
    max_ms = 0
    min_ms_not_0 = np.inf
    for j, (ms, pmz, tics, rtime) in enumerate(train_loader):
        local_max = ms.max()
        local_min = ms[ms > 0].min()
        if local_max > max_ms:
            max_ms = local_max
        if local_min < min_ms_not_0:
            min_ms_not_0 = local_min
        if mean is None:
            mean = ms.mean(0)
        else:
            mean += ms.mean(0)

    mean /= (len(train_loader) * 8)

    plt.figure(figsize=(20, 20))
    plt.plot(mean, color='r', marker='x', markersize=3, linewidth=0)
    plt.title(f"Mean ms")
    plt.xlabel('m/z')
    plt.ylabel('Normalized intensity')
    plt.savefig(f"stats/{label[0]}/mean_log{is_log}_norm{is_norm}_ms")
    plt.close()
    print(f"max log {is_log} norm {is_norm}: ", max_ms)
    print(f"min not zero, log {is_log} norm {is_norm}: ", min_ms_not_0)

    return mean


def std_ms(train_loader, label, mean, is_log, is_norm):
    var = None
    for j, (ms, pmz, tics, rtime) in enumerate(train_loader):
        for m in ms:
            if var is None:
                var = (m - mean)**2
            else:
                var += (m - mean)**2

    var /= (len(train_loader) * 8)
    std = np.sqrt(var)
    os.makedirs('stats/torch', exist_ok=True)
    torch.save(mean, f"stats/torch/means_{label[0]}")
    torch.save(std, f"stats/torch/stds_{label[0]}")

    plt.figure(figsize=(20, 20))
    plt.errorbar(list(range(len(mean))), mean.view(-1).detach().cpu().numpy(), std.view(-1).detach().cpu().numpy(), linewidth=0, elinewidth=1)
    plt.plot(mean.view(-1).detach().cpu().numpy(), color='r', marker='x', markersize=3, linewidth=0)
    plt.title(f"Mean ms")
    plt.xlabel('m/z')
    plt.ylabel('Normalized intensity')
    plt.savefig(f"stats/{label[0]}/mean-std_log{is_log}_norm{is_norm}_ms")
    plt.close()

    sem = std / np.sqrt(len(train_loader) * 8)
    plt.figure(figsize=(20, 20))
    plt.errorbar(list(range(len(mean))), mean.view(-1).detach().cpu().numpy(), sem.view(-1).detach().cpu().numpy(), linewidth=0, elinewidth=1)
    plt.plot(mean.view(-1).detach().cpu().numpy(), color='r', marker='x', markersize=3, linewidth=0)
    plt.title(f"Mean ms")
    plt.xlabel('m/z')
    plt.ylabel('Normalized intensity')
    plt.savefig(f"stats/{label[0]}/mean-sem_log{is_log}_norm{is_norm}_ms")
    plt.close()


def all_means_ms(meta_train_loader):
    mean = None
    max_ms = 0
    max_ms_array = None
    min_ms_not_0 = np.inf
    for i, (mss, label, colnames) in enumerate(meta_train_loader):
        train_set = MSDataset(mss.squeeze(), colnames, transform=None, log=False)
        train_loader = DataLoader(train_set,
                                  num_workers=0,
                                  shuffle=True,
                                  batch_size=8,
                                  pin_memory=True,
                                  drop_last=True)
        for j, (ms, pmz, tics, rtime) in enumerate(train_loader):
            for m in ms:
                if max_ms_array is None:
                    max_ms_array = m
                else:
                    max_ms_array = torch.Tensor([x.item() if x > m else m.item() for (x, m) in zip(m, max_ms_array)])
            local_max = ms.max()
            local_min = ms[ms > 0].min()
            if local_max > max_ms:
                max_ms = local_max
            if local_min < min_ms_not_0:
                min_ms_not_0 = local_min
            if mean is None:
                mean = ms.mean(0)
            else:
                mean += ms.mean(0)

    mean /= (len(train_loader) * 8)
    torch.save(max_ms_array, 'stats/torch/all_max')
    return mean


def all_stds_ms(meta_train_loader, mean):
    var = None
    for i, (mss, label, colnames) in enumerate(meta_train_loader):
        train_set = MSDataset(mss.squeeze(), colnames, transform=None, log=False)
        train_loader = DataLoader(train_set,
                                  num_workers=0,
                                  shuffle=True,
                                  batch_size=8,
                                  pin_memory=True,
                                  drop_last=True)
        for j, (ms, pmz, tics, rtime) in enumerate(train_loader):
            for m in ms:
                if var is None:
                    var = (m - mean) ** 2
                else:
                    var += (m - mean) ** 2

    var /= (len(train_loader) * 8)
    std = np.sqrt(var)
    os.makedirs('stats/torch', exist_ok=True)
    torch.save(mean, f"stats/torch/all_means")
    torch.save(std, f"stats/torch/all_stds")

    plt.figure(figsize=(20, 20))
    plt.errorbar(list(range(len(mean))), mean.view(-1).detach().cpu().numpy(), std.view(-1).detach().cpu().numpy(), linewidth=0, elinewidth=1)
    plt.plot(mean.view(-1).detach().cpu().numpy(), color='r', marker='x', markersize=3, linewidth=0)
    plt.title(f"Mean ms")
    plt.xlabel('m/z')
    plt.ylabel('Normalized intensity')
    plt.savefig(f"stats/{label[0]}/all_mean-std_logFalse_normFalse_ms")
    plt.close()

    sem = std / np.sqrt(len(train_loader) * 8)
    plt.figure(figsize=(20, 20))
    plt.errorbar(list(range(len(mean))), mean.view(-1).detach().cpu().numpy(), sem.view(-1).detach().cpu().numpy(), linewidth=0, elinewidth=1)
    plt.plot(mean.view(-1).detach().cpu().numpy(), color='r', marker='x', markersize=3, linewidth=0)
    plt.title(f"Mean ms")
    plt.xlabel('m/z')
    plt.ylabel('Normalized intensity')
    plt.savefig(f"stats/{label[0]}/all_mean-sem_logFalse_normFalse_ms")
    plt.close()



class Train:
    def __init__(self,
                 mss_path_train,
                 mss_path_valid,
                 bin='1'
                 ):
        super().__init__()
        self.bin = bin
        self.mss_path_train = mss_path_train
        self.mss_path_valid = mss_path_valid

    def train(self):
        global tb_logging, results
        random.seed(42)
        np.random.seed(42)

        train_set = MetaMSDataset(self.mss_path_train)
        valid_set = MetaMSDataset(self.mss_path_valid)

        meta_train_loader = DataLoader(train_set,
                                       num_workers=0,
                                       shuffle=True,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)
        meta_valid_loader = DataLoader(valid_set,
                                       num_workers=0,
                                       shuffle=True,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=False)

        # means = all_means_ms(meta_train_loader)
        # all_stds_ms(meta_train_loader, means)
        print('Processing individual samples...')
        for i, (mss, label, colnames) in enumerate(meta_train_loader):
            train_set = MSDataset(mss.squeeze(), colnames, transform=None, log=False)
            train(train_set, label, is_log=False, is_norm=False)
            train_set = MSDataset(mss.squeeze(), colnames, transform=None, log=True)
            train(train_set, label, is_log=True, is_norm=False)
            train_set = MSDataset(mss.squeeze(), colnames, log=True)
            train(train_set, label, is_log=True, is_norm=True)
            '''
            print(f"PrecursorMz: "
                  f"mean : {np.mean(pmz)}, ",
                  f"std : {np.std(pmz)}, ",
                  f"min : {np.min(pmz)}, ",
                  f"max : {np.max(pmz)}, ",
                  )
            print(f"tics: "
                  f"mean : {np.mean(tics)}, ",
                  f"std : {np.std(tics)}, ",
                  f"min : {np.min(tics)}, ",
                  f"max : {np.max(tics)}, ",
                  )
            print(f"rtime: "
                  f"mean : {np.mean(rtime)}, ",
                  f"std : {np.std(rtime)}, ",
                  f"min : {np.min(rtime)}, ",
                  f"max : {np.max(rtime)}, ",
                  )
            '''
        for i, (mss, label, colnames) in enumerate(meta_valid_loader):
            valid_set = MSDataset(mss.squeeze(), colnames, transform=None, log=False)
            train(valid_set, label, is_log=False, is_norm=False)
            valid_set = MSDataset(mss.squeeze(), colnames, transform=None, log=True)
            train(valid_set, label, is_log=True, is_norm=False)
            valid_set = MSDataset(mss.squeeze(), colnames, log=True)
            train(valid_set, label, is_log=True, is_norm=True)
            '''
            print(f"PrecursorMz: "
                  f"mean : {np.mean(pmz)}, ",
                  f"std : {np.std(pmz)}, ",
                  f"min : {np.min(pmz)}, ",
                  f"max : {np.max(pmz)}, ",
                  )
            print(f"tics: "
                  f"mean : {np.mean(tics)}, ",
                  f"std : {np.std(tics)}, ",
                  f"min : {np.min(tics)}, ",
                  f"max : {np.max(tics)}, ",
                  )
            print(f"rtime: "
                  f"mean : {np.mean(rtime)}, ",
                  f"std : {np.std(rtime)}, ",
                  f"min : {np.min(rtime)}, ",
                  f"max : {np.max(rtime)}, ",
                  )
            '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mss_path_train", type=str, help="Path to intensities csv file",
                        default='inputs_bin1/g30/m2/train_numpy')
    parser.add_argument("--mss_path_valid", type=str, help="Path to intensities csv file",
                        default='inputs_bin1/g30/m2/valid_numpy')
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--bin", type=str, default='1')
    args = parser.parse_args()

    training = Train(
        mss_path_train=args.mss_path_train,
        mss_path_valid=args.mss_path_valid,
        bin=args.bin
    )
    training.train()
