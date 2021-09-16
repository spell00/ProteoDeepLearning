#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import os
import pandas as pd
import numpy as np
import torch.nn as nn

from src.utils.utils import split_train_test

from src.utils.dataset import ms_data, get_normalized
from src.models.sklearn.ordination import pca
import matplotlib.pyplot as plt
import torch
from src.models.pytorch.utils.checkpoints import load_checkpoint
from sklearn.preprocessing import minmax_scale
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def get_optimizer(model, params):
    if params['optimizer_type'] == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=params['lr'],
                                     weight_decay=params['wd']
                                     )
    elif params['optimizer_type'] == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=params['lr'],
                                    weight_decay=params['wd'],
                                    momentum=0.9)
    elif params['optimizer_type'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(params=model.parameters(),
                                        lr=params['lr'],
                                        weight_decay=params['wd'],
                                        momentum=0.9)
    else:
        exit('error: no such optimizer type available')
    return optimizer


# TODO validate, display valid data differently to see if they look distant
def PCA(get_data_function, gen, args):
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
    gen_data = []
    for _ in range(int(len(labelss)/int(params['bs']))):
        noise = torch.randn([int(params['bs']), 64, 1]).to(device)
        gen_sample = gen(noise)
        gen_data += [gen_sample.reshape([-1, gen_sample.shape[-1]]).detach().cpu().numpy()]
        del noise, gen_sample

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    pca(datas, labelss, ax)
    os.makedirs("results/images/", exist_ok=True)
    plt.savefig(fname=f"results/images/pca_{args.g}_sample.png", dpi=100)
    plt.close()

    gen_data = np.stack(gen_data)
    gen_data = gen_data.reshape([-1, gen_data.shape[-1]])
    random_data = np.random.uniform(0, np.max(gen_data), gen_data.shape)
    # gen_data = minmax_scale(gen_data)
    labels_gen = np.array(['gen' for _ in gen_data])
    labels_random = np.array(['random' for _ in random_data])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    pca(gen_data, labels_gen, ax)
    os.makedirs("results/images/", exist_ok=True)
    plt.savefig(fname=f"results/images/pca_{args.g}_gen_pca.png", dpi=100)
    plt.close()

    labelss = np.concatenate((labelss, labels_gen), axis=0)
    datas = np.concatenate((datas[:, :gen_data.shape[-1]], gen_data), axis=0)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    pca(datas, labelss, ax)
    os.makedirs("results/images/", exist_ok=True)
    plt.savefig(fname=f"results/images/pca_{args.g}_samplegen_pca.png", dpi=100)
    plt.close()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    pca(datas, labelss, ax)
    # pca(gen_data, labels_gen, ax)
    os.makedirs("results/images/", exist_ok=True)
    plt.ylim([-0.011, 0.011])
    plt.xlim([-0.011, 0.011])
    plt.savefig(fname=f"results/images/pca_{args.g}_samplegen_zoom_pca.png", dpi=100)
    plt.close()

    labelss = np.concatenate((labelss, labels_gen, labels_random), axis=0)
    datas = np.concatenate((datas[:, :gen_data.shape[-1]], gen_data, random_data), axis=0)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    pca(datas, labelss, ax)
    os.makedirs("results/images/", exist_ok=True)
    plt.savefig(fname=f"results/images/pca_{args.g}_samplegenrandom_pca.png", dpi=100)
    plt.close()


def PCA2(fake_mss, mss, dir_name, epoch=-1):
    # labels = pd.read_csv(labels_file, header=0).loc[0]
    labels_dis = np.array(['real' for _ in mss])
    labels_gen = np.array(['gen' for _ in fake_mss])
    labels_random = np.array(['random' for _ in fake_mss])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    df = pca(mss, labels_dis, ax)
    os.makedirs(f"{dir_name}/pca_sample/", exist_ok=True)
    plt.savefig(fname=f"{dir_name}/pca_sample/{'{:08d}'.format(epoch)}.png", dpi=100)
    plt.close()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    df = pca(fake_mss, labels_gen, ax)
    os.makedirs(f"{dir_name}/pca_gen_pca/", exist_ok=True)
    plt.savefig(fname=f"{dir_name}/pca_gen_pca/{'{:08d}'.format(epoch)}.png", dpi=100)
    plt.close()

    labelss = np.concatenate((labels_dis, labels_gen), axis=0)
    datas = np.concatenate((mss, fake_mss), axis=0)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    df = pca(datas, labelss, ax)

    real_comps = df.iloc[np.where(df['label'] == 'real')[0], :-1]
    fake_comps = df.iloc[np.where(df['label'] == 'gen')[0], :-1]

    os.makedirs(f"{dir_name}/pca_samplegen_pca/", exist_ok=True)
    plt.savefig(fname=f"{dir_name}/pca_samplegen_pca/{'{:08d}'.format(epoch)}.png", dpi=100)
    plt.close()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    df = pca(datas, labelss, ax)
    # pca(gen_data, labels_gen, ax)
    os.makedirs(f"{dir_name}/pca_samplegen_zoom_pca/", exist_ok=True)
    plt.ylim([-0.011, 0.011])
    plt.xlim([-0.011, 0.011])
    plt.savefig(fname=f"{dir_name}/pca_samplegen_zoom_pca/{'{:08d}'.format(epoch)}.png", dpi=100)
    plt.close()

    labelss = np.concatenate((labelss, labels_random), axis=0)
    random_data = np.random.uniform(0, np.max(datas), mss.shape)
    datas = np.concatenate((datas, random_data), axis=0)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    df = pca(datas, labelss, ax)
    os.makedirs(f"{dir_name}/pca_samplegenrandom_pca/", exist_ok=True)
    plt.savefig(fname=f"{dir_name}/pca_samplegenrandom_pca/{'{:08d}'.format(epoch)}.png", dpi=100)
    plt.close()

    return real_comps, fake_comps


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--intensities_dir", type=str, default="inputs_bin1",
                        help="Path to intensities csv file")
    parser.add_argument("--mss_path_train", type=str, help="Path to intensities csv file",
                        default='inputs_bin1/g30/m2/train_numpy')
    parser.add_argument("--mss_path_valid", type=str, help="Path to intensities csv file",
                        default='inputs_bin1/g30/m2/valid_numpy')
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Checkpoint path")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--gen_out_activation", type=str, default='sigmoid')
    parser.add_argument("--dis_out_activation", type=str, default='linear')
    parser.add_argument("--model", type=str, default='pacgan4')
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--g", type=str, default='g30')
    parser.add_argument("--bin", type=str, default='1')
    parser.add_argument("--unroll_nsteps", type=int, default=3)
    parser.add_argument("--log1p", type=int, default=0)
    parser.add_argument("--norm", type=str, default='bn')
    parser.add_argument("--loss_type", type=str, default='mae')
    parser.add_argument("--n_gen", type=int, default=1)
    parser.add_argument("--n_dis", type=int, default=1)
    parser.add_argument("--load", type=int, default=0)
    args = parser.parse_args()
    if args.log1p == 0:
        args.log1p = False
    else:
        args.log1p = True

    if args.load == 0:
        args.load = False
    else:
        args.load = True

    params = {
        "max_value": np.nan,
        "bin": 1,
        "cv": 5,
        "bn": True,
        "bs": args.bs,
        "n_epochs": 10000,
        "size": 256,
        'epochs_per_checkpoint': 1,
        'optimizer_type': 'adam',
        'scheduler': "ReduceLROnPlateau",
        "res_blocks": False,
        "has_dense": True,
        "checkpoint_path": args.checkpoint_path,
        "z_dim": 100,
        "scale": 0.07,
        "n_res_channel": 64,  # TODO resnet not implemented
        "n_res": 18,  # TODO resnet not implemented
        "dropout": 0.2,
        "n_channel": 491,
        "n_pacgan": 4,
        "n_embed": 1012,
        "l1": 0,
        "wd": 1.039e-10,
        "lr": 5.643e-05
    }
    wd_str = str(params["wd"])
    l1_str = str(params["l1"])
    lr_str = str(params["lr"])
    wd_str = wd_str[:5] + wd_str[-4:]
    l1_str = l1_str[:5] + l1_str[-4:]
    lr_str = lr_str[:5] + lr_str[-4:]

    model_path = '/'.join([
        f"bin{params['bin']}",
        f"gen{args.gen_out_activation}",
        f"dis{args.dis_out_activation}",
        f"{params['scheduler']}",
        f'{args.loss_type}',
        f'{args.norm}',
        f'log{args.log1p}',
        f'{params["optimizer_type"]}',
        f'lr{lr_str}',
        f'nr{params["n_res"]}',
        f'z{params["z_dim"]}',
        f'ne{params["n_embed"]}',
        f'nrc{params["n_res_channel"]}',
        f'nc{params["n_channel"]}',
        f'wd{wd_str}',
        f'l1{l1_str}',
        f'd{params["dropout"]}',
        f's{params["scale"]}',
        f'unroll{args.unroll_nsteps}',
        f'pac{params["n_pacgan"]}',
    ])
    if 'dcgan' in args.model:
        from src.models.pytorch.unsupervised.GANs.DCGAN_1D import Generator as gen
        from src.models.pytorch.unsupervised.GANs.DCGAN_1D import Discriminator as dis
    elif 'pacgan' in args.model or 'pacwgan' in args.model:
        from src.models.pytorch.unsupervised.GANs.PacGAN_1D1 import Generator as gen
        from src.models.pytorch.unsupervised.GANs.PacGAN_1D1 import Discriminator as dis
    elif 'transgan' in args.model or 'transgan' in args.model:
        from src.models.pytorch.unsupervised.GANs.PacTransGAN_1D1 import Generator as gen
        from src.models.pytorch.unsupervised.GANs.PacTransGAN_1D1 import Discriminator as dis
    elif "gan" in args.model:
        from src.models.pytorch.unsupervised.GANs.GAN_1D import Generator as gen
        from src.models.pytorch.unsupervised.GANs.GAN_1D import Discriminator as dis
    else:
        exit('WRONG MODEL NAME')
    if args.gen_out_activation == 'sigmoid':
        gen = gen(
            out_activation=nn.Sigmoid,
            norm=args.norm
        ).to(device)
    elif args.gen_out_activation == 'linear':
        gen = gen(
            out_activation=None,
            norm=args.norm
        ).to(device)
    if args.dis_out_activation == 'sigmoid':
        dis = dis(
            out_activation=nn.Sigmoid,
            in_channel=params['n_pacgan'],
            norm=args.norm
        ).to(device)
    elif args.dis_out_activation == 'hardsigmoid':
        dis = dis(
            out_activation=nn.Hardsigmoid,
            in_channel=params['n_pacgan'],
            norm=args.norm
        ).to(device)
    elif args.dis_out_activation == 'linear':
        dis = dis(
            out_activation=None,
            in_channel=params['n_pacgan'],
            norm=args.norm
        ).to(device)

    optimizer_gen = get_optimizer(gen, params)
    optimizer_dis = get_optimizer(dis, params)

    gen, dis, optimizer_gen, optimizer_dis, epoch_offset = load_checkpoint(
        f"{args.checkpoint_path}/{model_path}",
        gen,
        dis,
        optimizer_gen,
        optimizer_dis,
        args.model
    )

    PCA(ms_data, gen, args)
