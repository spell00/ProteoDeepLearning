import os
import json
import torch
import random
import warnings
import datetime
import argparse
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from ax.service.managed_loop import optimize
from src.models.pytorch.GAN1D import VQVAE
from src.utils.CycleAnnealScheduler import CycleScheduler
from src.utils.dataset import load_checkpoint, save_checkpoint, MSDataset, MetaMSDataset

from src.utils.logging import TensorboardLogging

warnings.filterwarnings('ignore')


class DiceLoss(nn.Module):
    # source: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, y_true, y_preds, smooth=1):
        inputs = y_preds.view(-1)
        targets = y_true.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class JaccardLoss(nn.Module):
    # source: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
    def __init__(self, weight=None, size_average=True):
        super(JaccardLoss, self).__init__()

    def forward(self, y_true, y_preds, smooth=1):
        inputs = y_preds.view(-1)
        targets = y_true.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        jaccard = (intersection + smooth) / (union + smooth)

        return 1 - jaccard


# https://discuss.pytorch.org/t/how-to-implement-focal-loss-in-pytorch/6469
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=1):
        super().__init__()
        self.gamma = gamma

    def forward(self, log_pred_prob_onehot, target):
        pred_prob_oh = torch.exp(log_pred_prob_onehot)
        target = torch.tensor(target, dtype=torch.int64)
        pt = pred_prob_oh.data.gather(1, target)
        modulator = (1 - pt) ** self.gamma
        mce = modulator * (-torch.log(pt))

        return mce.mean()


class MCCLoss(nn.Module):
    def __init__(self):
        super(MCCLoss, self).__init__()

    def forward(self, y_true, y_preds):
        exit('not implemented')
        pass


dice_loss = DiceLoss()
jaccard_loss = JaccardLoss()
focal_loss = FocalLoss()


def compute_confusion_matrix(y_test, y_classes):
    from sklearn.metrics import confusion_matrix
    ys = y_test.view(-1).cpu().numpy()
    y_classes = y_classes.reshape(np.prod(y_classes.shape)).cpu().numpy()
    confusion_matrix = confusion_matrix(ys, y_classes)
    if sum(ys) == 0 and sum(y_classes) == 0:
        return 1, 1, 1, 1
    try:
        tn = confusion_matrix[0, 0]
        tp = confusion_matrix[1, 1]
        fp = confusion_matrix[0, 1]
        fn = confusion_matrix[1, 0]
    except:
        return np.nan, np.nan, np.nan, np.nan
    if tn != 0 and fp != 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 0
    if tp != 0 and fn != 0:
        sensitivity = tp / (tp + fn)
    else:
        sensitivity = 0
    if tp != 0 and fp != 0:
        precision = tp / (tp + fp)
        vdr = (fp - fn) / (tp + fn)
    else:
        precision = 0
        vdr = 0
    return sensitivity, specificity, precision, vdr


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


def print_results(results, mode, epoch):
    print(
        f"Epoch: {epoch}\t"
        f"{mode} Loss: {np.round(results['losses'][mode][-1], 4)} , "
        f"vol: {np.round(results['vol_target_ratio'][mode][-1], 4)} "
        f"vol bin: {np.round(results['vol_target_bin_ratio'][mode][-1], 4)} "
        f"kld: {np.round(results['kl_divs'][mode][-1], 4)} , "
        f"recon: {np.round(results['recon'][mode][-1], 4)}, "
        f"acc: {np.round(results['acc'][mode][-1], 4)} "
        f"Jaccard: {np.round(results['jaccard'][mode][-1], 4)}, "
        f"Dice: {np.round(results['dice'][mode][-1], 4)} "
        f"Sensitivity: {np.round(results['sensitivity'][mode][-1], 4)} "
        f"Specificity: {np.round(results['specificity'][mode][-1], 4)} "
        f"Precision: {np.round(results['precision'][mode][-1], 4)} "
    )


class Train:
    def __init__(self,
                 params,
                 model,
                 model_name,
                 images_path_train,
                 images_path_valid,
                 targets_path_train,
                 targets_path_valid,
                 loss,
                 checkpoint_path,
                 fp16_run=False,
                 epochs_per_checkpoint=1,
                 epochs_per_print=1,
                 verbose=2,
                 early_stop=100,
                 load=True,
                 save_checkpoints=True,
                 save=True,
                 ):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.fp16_run = fp16_run
        self.checkpoint_path = checkpoint_path
        self.epochs_per_checkpoint = epochs_per_checkpoint
        self.epochs_per_print = epochs_per_print
        self.save = save
        if loss == 'bce':
            self.criterion = nn.BCELoss()
        elif loss == 'focal':
            self.criterion = focal_loss
        elif loss == 'dice':
            self.criterion = dice_loss
        elif loss == 'jaccard':
            self.criterion = jaccard_loss
        else:
            exit(f'{loss} is not implemented. Valid choices are: [bce, dice, jaccard]')

        self.save_checkpoints = save_checkpoints
        self.load = load
        self.loss_type = loss
        self.verbose = verbose
        self.images_path_train = images_path_train
        self.images_path_valid = images_path_valid
        self.targets_path_train = targets_path_train
        self.targets_path_valid = targets_path_valid
        self.early_stop = early_stop
        self.params = params

    def print_params(self):
        print(
            f"Parameters: \n\t",
            # f'z_dim: {self.params["z_dim"]}\n\t',
            f'n_res: {self.params["n_res"]}\n\t',
            f'learning_rate: {self.params["lr"].__format__("e")} \n\t',
            f'weight_decay: {self.params["wd"].__format__("e")} \n\t',
            f'l1: {self.params["l1"].__format__("e")}\n\t',
            f'optimizer_type: {self.params["optimizer_type"]}\n\t',

        )

    def train(self, params_dict):
        global tb_logging, results
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        # TODO REMOVE THIS ; just for test when cpu is available, but not enough memory
        device = 'cpu'

        # n_res = params['n_res']
        self.params['z_dim'] = 90
        self.params['lr'] = params_dict['learning_rate']
        self.params['n_channel'] = params_dict['n_channel']
        self.params['n_res'] = params_dict['n_res']
        self.params['n_res_channel'] = params_dict['n_res_channel']
        self.params['scale'] = params_dict['scale']
        self.params['n_embed'] = params_dict['n_embed']
        self.params['wd'] = params_dict['weight_decay']
        self.params['l1'] = params_dict['l1']
        self.params['dropout'] = params_dict['dropout']

        # TODO have wd < 1e-8 equal 0 (or any other value like 1e-12). Same for l1
        self.params['l1'] = 0
        timestamp = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
        if self.verbose > 1:
            self.print_params()
        # Problem: find way to deal with different # of hparams
        model = self.model(
            in_channel=1,
            channel=self.params['n_channel'],
            n_res_block=self.params['n_res'],
            n_res_channel=self.params['n_res_channel'],
            embed_dim=self.params['z_dim'],
            n_embed=self.params['n_embed'],
            dropout=self.params['dropout']
        )

        model = self.model()

        wd_str = str(self.params["wd"])
        l1_str = str(self.params["l1"])
        lr_str = str(self.params["lr"])
        wd_str = wd_str[:5] + wd_str[-4:]
        l1_str = l1_str[:5] + l1_str[-4:]
        lr_str = lr_str[:5] + lr_str[-4:]
        scale_str = str(np.round(self.params['scale'], 2))
        drop_str = str(np.round(self.params['dropout'], 1))
        self.model_path = '/'.join([
            self.model_name,
            self.params['scheduler'],
            f'{self.loss_type}',
            f'{self.params["optimizer_type"]}',
            f'lr{lr_str}',
            f'nr{self.params["n_res"]}',
            f'z{self.params["z_dim"]}',
            f'ne{self.params["n_embed"]}',
            f'nrc{self.params["n_res_channel"]}',
            f'nc{self.params["n_channel"]}',
            f'wd{wd_str}',
            f'l1{l1_str}',
            f'd{drop_str}',
            f"s{scale_str}"
        ])
        # model.random_init()
        meta_optimizer = get_optimizer(model, self.params)

        # Load checkpoint if one exists
        epoch = 0
        best_dict = {
            'loss': -1,
            'dice': -1,
            'jaccard': -1,
            'sensitivity': -1,
            'specificity': -1,
            'accuracy': -1,
            'recon': -1,
        }
        if self.checkpoint_path is not None and self.load:
            try:
                model, optimizer, \
                epoch, losses, \
                best_dict['loss'] = load_checkpoint(checkpoint_path=self.checkpoint_path,
                                                    model=model,
                                                    params=self.params,
                                                    epoch=epoch,
                                                    predict=False,
                                                    optimizer=meta_optimizer,
                                                    name=self.model_name,
                                                    model_name=VQVAE,
                                                    timestamp=timestamp
                                                    )
            except IOError:
                print('No checkpoint found. Creating a new model.')
        meta_model = model.to(device)

        meta_train_set = MetaMSDataset(self.images_path_train, self.targets_path_train, transform=False)
        meta_valid_set = MetaMSDataset(self.images_path_valid, self.targets_path_valid, transform=False)
        # spliter = validation_spliter(all_set, cv=self.params['cv'])

        # TODO Make a test set

        epoch_offset = max(1, epoch + 1)

        # TODO replace with no CV (or cv=1 will be no CV?)
        for cv in range(1):
            meta_model.random_init()
            # best_dict['loss'] = -1
            # valid_set, train_set = spliter.__next__()

            os.makedirs('logs/' + self.model_path, exist_ok=True)
            hparams_filepath = 'logs/' + self.model_path + '/hparams'
            os.makedirs(hparams_filepath, exist_ok=True)
            logger = SummaryWriter('logs/' + self.model_path)
            tb_logging = TensorboardLogging(hparams_filepath, params_dict)

            results = {
                "losses": {
                    "train": [],
                    "valid": [],
                },
                "kl_divs": {
                    "train": [],
                    "valid": [],
                },
                "recon": {
                    "train": [],
                    "valid": [],
                },
                "abs_error": {
                    "train": [],
                    "valid": [],
                },
                "jaccard": {
                    "train": [],
                    "valid": [],
                },
                "dice": {
                    "train": [],
                    "valid": [],
                },
                "acc": {
                    "train": [],
                    "valid": [],
                },
                "vdr": {
                    "train": [],
                    "valid": [],
                },
                "specificity": {
                    "train": [],
                    "valid": [],
                },
                "sensitivity": {
                    "train": [],
                    "valid": [],
                },
                "precision": {
                    "train": [],
                    "valid": [],
                },
                "vol_target_ratio": {
                    "train": [],
                    "valid": [],
                },
                "vol_target_bin_ratio": {
                    "train": [],
                    "valid": [],
                },
            }

            early_stop_counter = 0
            print("\n\n\nCV:", cv, "/", self.params['cv'], "\nTrain samples:", len(meta_train_set),
                  "\nValid samples:", len(meta_valid_set), "\n\n\n")

            best_epoch = 0
            if epoch_offset == self.params['n_epochs']:
                return best_dict['loss']
            for epoch in range(epoch_offset, self.params['n_epochs']):
                is_best_epoch = False
                traces = {
                    "losses": {
                        "train": [],
                        "valid": [],
                    },
                    "kl_divs": {
                        "train": [],
                        "valid": [],
                    },
                    "recon": {
                        "train": [],
                        "valid": [],
                    },
                    "abs_error": {
                        "train": [],
                        "valid": [],
                    },
                    "acc": {
                        "train": [],
                        "valid": [],
                    },
                    "specificity": {
                        "train": [],
                        "valid": [],
                    },
                    "sensitivity": {
                        "train": [],
                        "valid": [],
                    },
                    "precision": {
                        "train": [],
                        "valid": [],
                    },
                    "vol_target_ratio": {
                        "train": [],
                        "valid": [],
                    },
                    "vol_target_bin_ratio": {
                        "train": [],
                        "valid": [],
                    },
                    "jaccard": {
                        "train": [],
                        "valid": [],
                    },
                    "dice": {
                        "train": [],
                        "valid": [],
                    },
                    # Volume difference rate
                    "vdr": {
                        "train": [],
                        "valid": [],
                    }
                }
                if early_stop_counter == self.early_stop:
                    if self.verbose > 0:
                        print('EARLY STOPPING.')
                    break
                meta_model.train()
                # pbar = tqdm(total=len(train_loader))
                meta_train_loader = DataLoader(meta_train_set,
                                               num_workers=0,
                                               shuffle=True,
                                               batch_size=self.params['bs'],
                                               pin_memory=True,
                                               drop_last=True)
                meta_valid_loader = DataLoader(meta_valid_set,
                                               num_workers=0,
                                               shuffle=True,
                                               batch_size=1,
                                               pin_memory=True,
                                               drop_last=False)
                meta_lr_scheduler = self.get_scheduler(meta_optimizer, len(meta_train_loader))
                m = model.to(device)

                for i, batch in enumerate(meta_train_loader):
                    images = batch
                    # TODO images and targets should be dataframes, not lists...
                    images = torch.Tensor(images.float()).to(device)
                    # targets = torch.Tensor(targets.float()).to(device)
                    train_set = MSDataset(images)

                    train_loader = DataLoader(train_set,
                                              num_workers=0,
                                              shuffle=True,
                                              batch_size=self.params['bs'],
                                              pin_memory=True,
                                              drop_last=True)
                    optimizer = get_optimizer(model, self.params)
                    lr_scheduler = self.get_scheduler(optimizer, len(meta_train_loader))

                    for i, batch in enumerate(train_loader):
                        # pbar.update(1)
                        images = batch
                        m.zero_grad()
                        reconstruct, kl = m(images.unsqueeze(0))

                        loss_recon = self.criterion(
                            reconstruct,  # .view(len(batch), -1),
                            images  # .view(len(batch), -1)
                        ).sum() / self.params['bs']
                        kl_div = torch.mean(kl)
                        loss = loss_recon + kl_div
                        if self.params['l1'] > 0:
                            l1 = self.get_l1(m, device)
                            loss += l1
                        loss.backward()

                        total = len(targets[0].view(-1))
                        acc = np.mean([torch.sum(x == y).item() for x, y in zip(bin_reconstruct, images)]) / total

                        sensitivity, specificity, precision, vdr = compute_confusion_matrix(y_test=images,
                                                                                            y_classes=bin_reconstruct)

                        traces['sensitivity']['train'] += [sensitivity]
                        traces['specificity']['train'] += [specificity]
                        traces['precision']['train'] += [precision]
                        traces['acc']['train'] += [acc]
                        traces['losses']['train'] += [loss.item()]
                        traces['kl_divs']['train'] += [kl_div.item()]
                        traces['recon']['train'] += [loss_recon.item()]
                        traces['dice']['train'] += [dice.item()]
                        traces['jaccard']['train'] += [jaccard.item()]
                        traces['vdr']['train'] += [vdr]
                        traces["vol_target_ratio"]['train'] += [vol_target_ratio.item()]
                        traces["vol_target_bin_ratio"]['train'] += [vol_target_bin_ratio.item()]
                        traces['abs_error']['train'] += [
                            float(torch.mean(torch.abs_(
                                reconstruct - images.to(device)
                            )).item())
                        ]

                        optimizer.step()
                        if self.params['scheduler'] == "CycleScheduler":
                            lr_scheduler.step()
                            # optimizer = lr_scheduler.optimizer
                        del kl, loss_recon, kl_div, loss, dice

                results["specificity"]["train"] += [np.mean(traces['specificity']['train'])]
                results["sensitivity"]["train"] += [np.mean(traces['sensitivity']['train'])]
                results["precision"]["train"] += [np.mean(traces['precision']['train'])]
                results["losses"]["train"] += [np.mean(traces['losses']['train'])]
                results["kl_divs"]["train"] += [np.mean(traces['kl_divs']['train'])]
                results["recon"]["train"] += [np.mean(traces['recon']['train'])]
                results["abs_error"]["train"] += [np.mean(traces['abs_error']['train'])]
                results["acc"]["train"] += [np.mean(traces['acc']['train'])]
                results["jaccard"]["train"] += [np.mean(traces['jaccard']['train'])]
                results['vdr']['train'] += [np.mean(traces['vdr']['train'])]
                results["dice"]["train"] += [np.mean(traces['dice']['train'])]
                results["vol_target_ratio"]["train"] += [np.mean(traces["vol_target_ratio"]['train'])]
                results["vol_target_bin_ratio"]["train"] += [np.mean(traces["vol_target_bin_ratio"]['train'])]

                logger.add_scalar('train/specificity', results["specificity"]["train"][-1], epoch)
                logger.add_scalar('train/sensitivity', results["sensitivity"]["train"][-1], epoch)
                logger.add_scalar('train/precision', results["precision"]["train"][-1], epoch)
                logger.add_scalar('train/loss', results["losses"]["train"][-1], epoch)
                if results["kl_divs"]["train"][-1] > 0:
                    logger.add_scalar('train/kld', results["kl_divs"]["train"][-1], epoch)
                logger.add_scalar('train/recon', results["recon"]["train"][-1], epoch)
                logger.add_scalar('train/accuracy', results["acc"]["train"][-1], epoch)
                logger.add_scalar('train/jaccard', results["jaccard"]["train"][-1], epoch)
                logger.add_scalar('train/dice', results["dice"]["train"][-1], epoch)
                logger.add_scalar('train/vdr', results["vdr"]["train"][-1], epoch)
                logger.add_scalar('train/vol', results["vol_target_ratio"]["train"][-1], epoch)
                logger.add_scalar('train/vol_bin', results["vol_target_bin_ratio"]["train"][-1], epoch)

                if epoch % self.epochs_per_print == 0:
                    if self.verbose > 1:
                        print_results(results, mode='train', epoch=epoch)

                m.eval()
                with torch.no_grad():
                    for i, batch in enumerate(valid_loader):
                        images3d, targets3d = batch
                        new_image3d = torch.zeros_like(images3d[0])
                        new_target3d = torch.zeros_like(targets3d[0])
                        reconstruct3d = torch.zeros_like(targets3d[0])
                        bin_reconstruct3d = torch.zeros_like(targets3d[0])
                        for j in range(images3d.shape[-1]):
                            images = images3d[:, :, :, j].to(device)
                            targets = targets3d[:, :, :, j].to(device)
                            try:
                                reconstruct, kl = m(images.unsqueeze(1))
                            except:
                                continue
                            bin_reconstruct = (reconstruct > 0.5).float()

                            volume_target = targets.sum()
                            volume_rec = reconstruct.sum()
                            volume_rec_bin = bin_reconstruct.sum()
                            vol_target_ratio = volume_target / volume_rec
                            vol_target_bin_ratio = volume_target / volume_rec_bin
                            if torch.isinf(vol_target_ratio) or torch.isnan(vol_target_ratio):
                                vol_target_ratio = torch.Tensor([1])
                            if torch.isinf(vol_target_bin_ratio) or torch.isnan(vol_target_bin_ratio):
                                vol_target_bin_ratio = torch.Tensor([1])
                            loss_recon = self.criterion(
                                reconstruct.view(1, -1),
                                targets.view(1, -1)
                            ).sum()
                            kl_div = torch.mean(kl)
                            dice = dice_loss(y_true=targets, y_preds=bin_reconstruct)
                            loss = loss_recon + kl_div
                            jaccard = jaccard_loss(
                                y_true=targets.reshape(targets.shape[0], -1),
                                y_preds=bin_reconstruct.reshape(targets.shape[0], -1)
                            )

                            sensitivity, specificity, precision, vdr = compute_confusion_matrix(y_test=targets,
                                                                                                y_classes=bin_reconstruct)

                            total = len(targets.view(-1))
                            acc = np.mean([torch.sum(x == y).item() for x, y in zip(bin_reconstruct, targets)]) / total

                            traces['sensitivity']['valid'] += [sensitivity]
                            traces['specificity']['valid'] += [specificity]
                            traces['precision']['valid'] += [precision]
                            traces['acc']['valid'] += [acc]
                            traces['losses']['valid'] += [loss.item()]
                            traces['kl_divs']['valid'] += [kl_div.item()]
                            traces['jaccard']['valid'] += [jaccard.item()]
                            traces['dice']['valid'] += [dice.item()]
                            traces['vdr']['valid'] += [vdr]
                            traces["vol_target_ratio"]['valid'] += [vol_target_ratio.item()]
                            traces["vol_target_bin_ratio"]['valid'] += [vol_target_bin_ratio.item()]
                            traces['recon']['valid'] += [loss_recon.item()]
                            traces['abs_error']['valid'] += [
                                float(torch.mean(torch.abs_(reconstruct - images.to(device))).item())]
                            new_image3d[:, :, j] = images.squeeze()
                            new_target3d[:, :, j] = targets.squeeze()
                            reconstruct3d[:, :, j] = reconstruct.squeeze()
                            bin_reconstruct3d[:, :, j] = bin_reconstruct.squeeze()
                results["specificity"]["valid"] += [np.mean(traces['specificity']['valid'])]
                results["sensitivity"]["valid"] += [np.mean(traces['sensitivity']['valid'])]
                results["precision"]["valid"] += [np.mean(traces['precision']['valid'])]
                results['losses']["valid"] += [np.mean(traces['losses']['valid'])]
                results['kl_divs']["valid"] += [np.mean(traces['kl_divs']['valid'])]
                results['recon']["valid"] += [np.mean(traces['recon']['valid'])]
                results['abs_error']["valid"] += [np.mean(traces['abs_error']['valid'])]
                results['acc']["valid"] += [np.mean(traces['acc']['valid'])]
                results['jaccard']["valid"] += [np.mean(traces['jaccard']['valid'])]
                results['vdr']["valid"] += [np.mean(traces['vdr']['valid'])]
                results['dice']["valid"] += [np.mean(traces['dice']['valid'])]
                results["vol_target_ratio"]["valid"] += [np.mean(traces["vol_target_ratio"]['valid'])]
                results["vol_target_bin_ratio"]["valid"] += [np.mean(traces["vol_target_bin_ratio"]['valid'])]

                logger.add_scalar('valid/specificity', results["specificity"]["valid"][-1], epoch)
                logger.add_scalar('valid/sensitivity', results["sensitivity"]["valid"][-1], epoch)
                logger.add_scalar('valid/precision', results["precision"]["valid"][-1], epoch)
                logger.add_scalar('valid/loss', results['losses']['valid'][-1], epoch)
                if results["kl_divs"]["valid"][-1] > 0:
                    logger.add_scalar('valid/kld', results['kl_divs']['valid'][-1], epoch)
                logger.add_scalar('valid/recon', results['recon']['valid'][-1], epoch)
                logger.add_scalar('valid/acc', results['acc']['valid'][-1], epoch)
                logger.add_scalar('valid/jaccard', results['jaccard']['valid'][-1], epoch)
                logger.add_scalar('valid/vdr', results['vdr']['valid'][-1], epoch)
                logger.add_scalar('valid/dice', results['dice']['valid'][-1], epoch)
                logger.add_scalar('valid/vol', results["vol_target_ratio"]['valid'][-1], epoch)
                logger.add_scalar('valid/vol_bin', results["vol_target_bin_ratio"]['valid'][-1], epoch)
                if self.params['scheduler'] == "ReduceLROnPlateau":
                    if epoch - epoch_offset > 5:
                        lr_scheduler.step(results['losses']["valid"][-1])
                if (results['losses']['valid'][-1] < best_dict['loss'] or best_dict['loss'] == -1) and not np.isnan(
                        results['losses']['valid'][-1]):
                    if self.verbose > 1:
                        print('BEST EPOCH!', results['losses']['valid'][-1])
                    early_stop_counter = 0
                    best_dict['loss'] = results['losses']['valid'][-1]
                    best_dict['dice'] = results['dice']['valid'][-1]
                    best_dict['jaccard'] = results['jaccard']['valid'][-1]
                    best_dict['vdr'] = results['vdr']['valid'][-1]
                    best_dict['sensitivity'] = results['sensitivity']['valid'][-1]
                    best_dict['specificity'] = results['specificity']['valid'][-1]
                    best_dict['recon'] = results['recon']['valid'][-1]
                    is_best_epoch = True
                    best_epoch = epoch
                else:
                    early_stop_counter += 1

                if epoch % self.epochs_per_checkpoint == 0:
                    if self.save:
                        if f'views/{self.model_path}' not in os.listdir():
                            os.makedirs(f'views/{self.model_path}', exist_ok=True)

                        logger.add_image(tag='valid/target', img_tensor=targets.detach().cpu(),
                                         global_step=epoch)
                        logger.add_image(tag='valid/reconstruction', img_tensor=reconstruct.detach().cpu()[0],
                                         global_step=epoch)
                        logger.add_image(tag='valid/bin_reconstruction', img_tensor=bin_reconstruct.detach().cpu()[0],
                                         global_step=epoch)
                        logger.add_image(tag='valid/input', img_tensor=images.detach().cpu(),
                                         global_step=epoch)
                    if is_best_epoch and self.save_checkpoints:
                        if self.verbose > 1:
                            print('Saving model...')
                        losses_to_save = {
                            'losses': results['losses'],
                            'kl_divs': results['kl_divs'],
                            'recon': results['recon'],
                        }

                        save_checkpoint(
                            model=m,
                            optimizer=optimizer,
                            epoch=epoch,
                            checkpoint_path=self.checkpoint_path,
                            losses=losses_to_save,
                            name=self.model_path,
                            model_name=self.model_name
                        )
                if epoch % self.epochs_per_print == 0:
                    if self.verbose > 0:
                        print_results(results, mode='valid', epoch=epoch)

                    if self.verbose > 1:
                        print("Current LR:", optimizer.param_groups[0]['lr'])
                    if 'momentum' in optimizer.param_groups[0].keys():
                        print("Current Momentum:", optimizer.param_groups[0]['momentum'])
            if self.verbose > 0:
                print('BEST LOSS :', best_dict['loss'])

        tb_logging.logging(results)

        return best_dict['jaccard']

    def get_scheduler(self, optimizer, n_samples=None):
        if self.params['scheduler'] == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                      factor=0.1,
                                                                      cooldown=0,
                                                                      patience=5,
                                                                      verbose=True,
                                                                      min_lr=1e-8)
        elif self.params['scheduler'] == 'CycleScheduler':
            assert n_samples is not None
            lr_scheduler = CycleScheduler(optimizer,
                                          self.params['lr'],
                                          n_iter=self.params['n_epochs'] * n_samples)
        else:
            lr_scheduler = None
        return lr_scheduler

    def get_l1(self, model, device):
        l1_reg = torch.tensor(0.).to(device)
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1_reg += torch.norm(param, 1).to(device)

        return self.params['l1'] * l1_reg

    def get_volume(self):
        pass


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path_train", type=str, help="Path to intensities csv file",
                        default='inputs_bin01/g30/train_numpy')
    parser.add_argument("--images_path_valid", type=str, help="Path to intensities csv file",
                        default='inputs_bin01/g30/valid_numpy')
    parser.add_argument("--labels_path_train", type=str, help="Path to labels csv file",
                        default='inputs_bin01/g90/train_numpy')
    parser.add_argument("--labels_path_valid", type=str, help="Path to labels csv file",
                        default='inputs_bin01/g90/valid_numpy')
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Checkpoint path")
    parser.add_argument("--loss", type=str, default="bce", help="Path to labels csv file")
    parser.add_argument("--verbose", type=str, default=1)
    parser.add_argument("--model", type=str, default='vqvae')
    args = parser.parse_args()

    if args.model == 'vqvae':
        from src.models.pytorch.unsupervised.VAEs.VQVAE2_1D import VQVAE as model
    elif args.model == "unet":
        from src.models.pytorch.unsupervised.VAEs.unet import UNet as model
    elif args.model == "fcn":
        exit('Not implemented')
    elif args.model == "voxnet":
        exit('Not implemented')
        # from fmri.models.unsupervised.voxnet2d import VoxNet2d as model
    else:
        exit('WRONG MODEL NAME')

    params = {
        "cv": 5,
        "bn": True,
        "bs": 1,
        "n_epochs": 1000,
        "size": 256,
        'epochs_per_checkpoint': 1,
        'optimizer_type': 'adam',
        'scheduler': "ReduceLROnPlateau",
        "res_blocks": True,
        "has_dense": True,
        # All params that are None are optimized with Bayesian process
        "z_dim": None,
        "dropout": None,
        "lr": None,
        "wd": None,
        "n_res": None,
        "l1": None,
        'n_channel': None,
        'n_embed': None,
        "n_res_channel": None,
    }

    training = Train(
        params,
        model,
        images_path_train=args.images_path_train,
        images_path_valid=args.images_path_valid,
        targets_path_train=args.labels_path_train,
        targets_path_valid=args.labels_path_valid,
        loss=args.loss,
        checkpoint_path=args.checkpoint_path,
        save=True,
        load=False,
        early_stop=25,
        save_checkpoints=True,
        model_name=args.model

    )

    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "scale", "type": "range", "bounds": [0., 0.1]},
            # {"name": "z_dim", "type": "range", "bounds": [2, 64]},
            {"name": "n_res", "type": "range", "bounds": [10, 20]},
            {"name": "dropout", "type": "range", "bounds": [0., 0.5]},
            {"name": "n_channel", "type": "range", "bounds": [256, 512]},
            {"name": "n_embed", "type": "range", "bounds": [512, 1024]},
            {"name": "n_res_channel", "type": "range", "bounds": [32, 64]},
            {"name": "l1", "type": "range", "bounds": [1e-12, 1e-3], "log_scale": True},
            {"name": "weight_decay", "type": "range", "bounds": [1e-12, 1e-3], "log_scale": True},
            {"name": "learning_rate", "type": "range", "bounds": [1e-4, 1e-3], "log_scale": True},
        ],
        evaluation_function=training.train,
        objective_name='loss',
        minimize=True,
        total_trials=100
    )
    from matplotlib import pyplot as plt

    fig = plt.figure()
    # render(plot_contour(model=model, param_x="learning_rate", param_y="weight_decay", metric_name='Loss'))
    # fig.savefig('test.jpg')
    print('Best Loss:', values[0]['loss'])
    print('Best Parameters:')
    print(json.dumps(best_parameters, indent=4))

    # cv_results = cross_validate(model)
    # render(interact_cross_validation(cv_results))
