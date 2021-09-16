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
from src.models.pytorch.unsupervised.VAEs.VQVAE_1D import VariationalAutoencoderVQ as VQVAE
from src.utils.CycleAnnealScheduler import CycleScheduler
from src.utils.dataset import load_checkpoint, save_checkpoint, MSDataset, MetaMSDataset

from src.utils.logging import TensorboardLogging

warnings.filterwarnings('ignore')



class MCCLoss(nn.Module):
    def __init__(self):
        super(MCCLoss, self).__init__()

    def forward(self, y_true, y_preds):
        exit('not implemented')
        pass


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
        f"{mode} Loss: {np.round(results['losses'][mode][-1], 8)} , "
        # f"vol: {np.round(results['vol_target_ratio'][mode][-1], 4)} "
        # f"vol bin: {np.round(results['vol_target_bin_ratio'][mode][-1], 4)} "
        f"kld: {np.round(results['kl_divs'][mode][-1], 8)} , "
        f"recon: {np.round(results['recon'][mode][-1], 8)}, "
        # f"acc: {np.round(results['acc'][mode][-1], 4)} "
        # f"Jaccard: {np.round(results['jaccard'][mode][-1], 4)}, "
        # f"Dice: {np.round(results['dice'][mode][-1], 4)} "
        # f"Sensitivity: {np.round(results['sensitivity'][mode][-1], 4)} "
        # f"Specificity: {np.round(results['specificity'][mode][-1], 4)} "
        # f"Precision: {np.round(results['precision'][mode][-1], 4)} "
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
                 save=False,
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
        elif loss == 'mse':
            self.criterion = nn.MSELoss()
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
            f'z_dim: {self.params["z_dim"]}\n\t',
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
        # device = 'cpu'

        # n_res = params['n_res']
        self.params['z_dim'] = 100
        self.params['lr'] = params_dict['learning_rate']
        self.params['n_channel'] = params_dict['n_channel']
        self.params['n_res'] = params_dict['n_res']
        self.params['n_res_channel'] = params_dict['n_res_channel']
        self.params['scale'] = params_dict['scale']
        self.params['n_embed'] = params_dict['n_embed']
        self.params['wd'] = params_dict['weight_decay']
        self.params['l1'] = params_dict['l1']
        self.params['dropout'] = params_dict['dropout']

        in_channels = [1, 32, 64, 128, 256, 512]
        out_channels = [32, 64, 128, 256, 512, 1]
        # kernel_sizes = [5, 5, 5, 5, 5, 5, 1]
        kernel_sizes = [15, 3, 3, 3, 3, 1]
        strides = [2, 2, 2, 2, 2, 1]
        dilatations = [1, 2, 4, 8, 16, 32]

        # TODO have wd < 1e-8 equal 0 (or any other value like 1e-12). Same for l1
        self.params['l1'] = 0
        timestamp = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
        if self.verbose > 1:
            self.print_params()
        # Problem: find way to deal with different # of hparams
        if self.model_name == "vqvae":
            model = self.model(
                self.params['z_dim'],
                in_channels,
                out_channels,
                kernel_sizes,
                strides,
                dilatations,
                256
            )

        elif self.model_name in ["unet", "voxnet"]:
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
        optimizer = get_optimizer(model, self.params)

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
                                                    optimizer=optimizer,
                                                    name=self.model_name,
                                                    model_name=VQVAE,
                                                    timestamp=timestamp
                                                    )
            except IOError:
                print('No checkpoint found. Creating a new model.')
        model = model.to(device)

        train_set = MetaMSDataset(self.images_path_train, self.targets_path_train, transform=False)
        valid_set = MetaMSDataset(self.images_path_valid, self.targets_path_valid, transform=False)
        # spliter = validation_spliter(all_set, cv=self.params['cv'])

        # TODO Make a test set

        epoch_offset = max(1, epoch + 1)

        # TODO replace with no CV (or cv=1 will be no CV?)
        for cv in range(1):
            model.random_init()
            # best_dict['loss'] = -1
            # valid_set, train_set = spliter.__next__()

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

            os.makedirs('logs/' + self.model_path, exist_ok=True)
            hparams_filepath = 'logs/' + self.model_path + '/hparams'
            os.makedirs(hparams_filepath, exist_ok=True)
            logger = SummaryWriter('logs/' + self.model_path)
            tb_logging = TensorboardLogging(hparams_filepath, params_dict)

            lr_scheduler = self.get_scheduler(optimizer, len(meta_train_loader))

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
                }
            }

            early_stop_counter = 0
            print("\n\n\nCV:", cv, "/", self.params['cv'], "\nTrain samples:", len(train_set),
                  "\nValid samples:", len(valid_set), "\n\n\n")

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
                    }
                }
                if early_stop_counter == self.early_stop:
                    if self.verbose > 0:
                        print('EARLY STOPPING.')
                    break
                model.train()
                # pbar = tqdm(total=len(train_loader))
                for i, images in enumerate(meta_train_loader):
                    train_set = MSDataset(images.squeeze())

                    train_loader = DataLoader(train_set,
                                              num_workers=0,
                                              shuffle=True,
                                              batch_size=self.params['bs'],
                                              pin_memory=True,
                                              drop_last=True)
                    for j, image in enumerate(train_loader):
                        # pbar.update(1)
                        model.zero_grad()
                        # images #, targets = batch
                        # TODO images and targets should be dataframes, not lists...
                        image = torch.Tensor(image.float()).to(device)
                        image = torch.cat((image, torch.zeros(image.shape[0], 5).to(device)), 1)
                        # targets = image
                        reconstruct, kl = model(image.unsqueeze(1))
                        loss_recon = self.criterion(
                            reconstruct.squeeze(1),  # .view(len(batch), -1),
                            image  # .view(len(batch), -1)
                        ) / self.params['bs']
                        kl_div = torch.mean(kl)
                        loss = loss_recon + kl_div
                        if self.params['l1'] > 0:
                            l1 = self.get_l1(model, device)
                            loss += l1
                        loss.backward()
                        traces['losses']['train'] += [loss.item()]
                        traces['kl_divs']['train'] += [kl_div.item()]
                        traces['recon']['train'] += [loss_recon.item()]

                        optimizer.step()
                        if self.params['scheduler'] == "CycleScheduler":
                            lr_scheduler.step()
                            # optimizer = lr_scheduler.optimizer
                        del kl, loss_recon, kl_div, loss  # , dice

                results["losses"]["train"] += [np.mean(traces['losses']['train'])]
                results["kl_divs"]["train"] += [np.mean(traces['kl_divs']['train'])]
                results["recon"]["train"] += [np.mean(traces['recon']['train'])]
                logger.add_scalar('train/loss', results["losses"]["train"][-1], epoch)
                if results["kl_divs"]["train"][-1] > 0:
                    logger.add_scalar('train/kld', results["kl_divs"]["train"][-1], epoch)
                logger.add_scalar('train/recon', results["recon"]["train"][-1], epoch)
                if epoch % self.epochs_per_print == 0:
                    if self.verbose > 1:
                        print_results(results, mode='train', epoch=epoch)

                model.eval()
                with torch.no_grad():
                    for i, images in enumerate(meta_valid_loader):
                        valid_set = MSDataset(images)

                        valid_loader = DataLoader(valid_set,
                                                  num_workers=0,
                                                  shuffle=True,
                                                  batch_size=self.params['bs'],
                                                  pin_memory=True,
                                                  drop_last=True)
                        for j, image in enumerate(valid_loader):
                            image = torch.Tensor(image.float()).to(device)
                            image = torch.cat((image, torch.zeros(image.shape[0], 5).to(device)), 1)
                            reconstruct, kl = model(image.unsqueeze(1))
                            targets = image
                            loss_recon = self.criterion(
                                reconstruct.squeeze(1),  # .view(len(batch), -1),
                                targets  # .view(len(batch), -1)
                            ) / self.params['bs']
                            kl_div = torch.mean(kl)
                            loss = loss_recon + kl_div
                            traces['losses']['valid'] += [loss.item()]
                            traces['kl_divs']['valid'] += [kl_div.item()]
                            traces['recon']['valid'] += [loss_recon.item()]

                results['losses']["valid"] += [np.mean(traces['losses']['valid'])]
                results['kl_divs']["valid"] += [np.mean(traces['kl_divs']['valid'])]
                results['recon']["valid"] += [np.mean(traces['recon']['valid'])]
                logger.add_scalar('valid/loss', results['losses']['valid'][-1], epoch)
                if results["kl_divs"]["valid"][-1] > 0:
                    logger.add_scalar('valid/kld', results['kl_divs']['valid'][-1], epoch)
                logger.add_scalar('valid/recon', results['recon']['valid'][-1], epoch)
                if self.params['scheduler'] == "ReduceLROnPlateau":
                    if epoch - epoch_offset > 5:
                        lr_scheduler.step(results['losses']["valid"][-1])
                if (results['losses']['valid'][-1] < best_dict['loss'] or best_dict['loss'] == -1) and not np.isnan(
                        results['losses']['valid'][-1]):
                    if self.verbose > 1:
                        print('BEST EPOCH!', results['losses']['valid'][-1])
                    early_stop_counter = 0
                    best_dict['loss'] = results['losses']['valid'][-1]
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
                        # logger.add_image(tag='valid/bin_reconstruction', img_tensor=bin_reconstruct.detach().cpu()[0],
                        #                  global_step=epoch)
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
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            checkpoint_path=self.checkpoint_path,
                            losses=losses_to_save,
                            name=f"{self.model_path}/vqvae.model",
                            learning_rate=self.params['lr'],
                            best_loss=None,
                            best_accuracy=None,
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
    parser.add_argument("--loss", type=str, default="mse", help="Path to labels csv file")
    parser.add_argument("--verbose", type=str, default=1)
    parser.add_argument("--model", type=str, default='vqvae')
    args = parser.parse_args()

    if args.model == 'vqvae2':
        from src.models.pytorch.unsupervised.VAEs.VQVAE2_1D import VQVAE as model
    elif args.model == 'vqvae':
        from src.models.pytorch.unsupervised.VAEs.VQVAE_1D import VariationalAutoencoderVQ as model
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
        "max_value": np.nan,
        "cv": 5,
        "bn": True,
        "bs": 3,
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
        save=False,
        load=False,
        early_stop=25,
        save_checkpoints=False,
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
            {"name": "learning_rate", "type": "range", "bounds": [1e-3, 1e-1], "log_scale": True},
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
