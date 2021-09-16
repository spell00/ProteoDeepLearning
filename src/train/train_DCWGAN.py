import os
import json
import torch
import random
import warnings
import datetime
import argparse
import numpy as np
import nibabel as nib
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from ax.service.managed_loop import optimize
from src.models.pytorch.GAN_1D import Generator, Discriminator
from src.utils.CycleAnnealScheduler import CycleScheduler
from src.utils.dataset import MSDataset, MetaMSDataset
from src.models.pytorch.utils.checkpoints import load_checkpoint, save_checkpoint
import torch.nn.functional as F
from src.utils.logging import TensorboardLogging
from matplotlib import pyplot as plt

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
        f"{mode} Loss Gen: {np.round(results['losses_gen'][mode][-1], 8)} , "
        f"{mode} Loss Dis Real: {np.round(results['losses_dis_real'][mode][-1], 8)} , "
        f"{mode} Loss Dis Fake: {np.round(results['losses_dis_fake'][mode][-1], 8)} , "
    )


class Train:
    def __init__(self,
                 params,
                 gen,
                 dis,
                 images_path_train,
                 images_path_valid,
                 targets_path_train,
                 targets_path_valid,
                 loss,
                 checkpoint_path,
                 model_name,
                 gen_out_activation,
                 dis_out_activation,
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
        self.gen_out_activation = gen_out_activation
        self.dis_out_activation = dis_out_activation
        self.fp16_run = fp16_run
        self.model_name = model_name
        if 'pacgan' in self.model_name:
            self.n_pacgan = int(self.model_name.split('pacgan')[1])
        elif 'pacwgan' in self.model_name:
            self.n_pacgan = int(self.model_name.split('pacwgan')[1])
        else:
            self.n_pacgan = None

        if gen_out_activation == 'sigmoid':
            self.gen = gen(out_activation=nn.Sigmoid)
        elif gen_out_activation == 'linear':
            self.gen = gen(out_activation=None)

        if dis_out_activation == 'sigmoid':
            self.dis = dis(out_activation=nn.Sigmoid,
                           in_channel=self.n_pacgan)
        elif dis_out_activation == 'hardsigmoid':
            self.dis = dis(out_activation=nn.Hardsigmoid,
                           in_channel=self.n_pacgan)
        elif dis_out_activation == 'linear':
            self.dis = dis(out_activation=None,
                           in_channel=self.n_pacgan)
        self.checkpoint_path = checkpoint_path
        self.epochs_per_checkpoint = epochs_per_checkpoint
        self.epochs_per_print = epochs_per_print
        self.save = save
        if loss == 'bce':
            self.criterion = nn.BCELoss()
            # self.criterion_d = nn.MSELoss()
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

        # TODO have wd < 1e-8 equal 0 (or any other value like 1e-12). Same for l1
        self.params['l1'] = 0
        timestamp = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
        if self.verbose > 1:
            self.print_params()
        # Problem: find way to deal with different # of hparams

        wd_str = str(self.params["wd"])
        l1_str = str(self.params["l1"])
        lr_str = str(self.params["lr"])
        wd_str = wd_str[:5] + wd_str[-4:]
        l1_str = l1_str[:5] + l1_str[-4:]
        lr_str = lr_str[:5] + lr_str[-4:]
        scale_str = str(np.round(self.params['scale'], 2))
        drop_str = str(np.round(self.params['dropout'], 1))
        self.model_path = '/'.join([
            f"gen{self.gen_out_activation}",
            f"dis{self.gen_out_activation}",
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

        optimizer_gen = get_optimizer(self.gen, self.params)
        optimizer_dis = get_optimizer(self.dis, self.params)

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

        gen = self.gen.to(device)
        dis = self.dis.to(device)

        train_set = MetaMSDataset(self.images_path_train, self.targets_path_train, transform=False)
        valid_set = MetaMSDataset(self.images_path_valid, self.targets_path_valid, transform=False)
        # spliter = validation_spliter(all_set, cv=self.params['cv'])

        # TODO Make a test set

        epoch_offset = max(1, epoch + 1)

        # TODO replace with no CV (or cv=1 will be no CV?)
        optimizer_gen.param_groups[0]['lr'] = self.params['lr']
        optimizer_dis.param_groups[0]['lr'] = self.params['lr']
        if self.load:
            gen, dis, optimizer_gen, optimizer_dis, epoch_offset = load_checkpoint(
                    f"{self.params['checkpoint_path']}/{self.model_path}",
                    gen,
                    dis,
                    optimizer_gen,
                    optimizer_dis,
                    self.model_name
                )
        print("Path: ", self.model_path)
        for cv in range(1):
            # gen.random_init()
            # dis.random_init()
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
            one = torch.FloatTensor([1])
            mone = one * -1

            os.makedirs('logs/' + self.model_path, exist_ok=True)
            hparams_filepath = 'logs/' + self.model_path + '/hparams'
            os.makedirs(hparams_filepath, exist_ok=True)
            logger = SummaryWriter('logs/' + self.model_path)
            tb_logging = TensorboardLogging(hparams_filepath, params_dict)

            lr_scheduler_gen = self.get_scheduler(optimizer_gen, 10000)
            lr_scheduler_dis = self.get_scheduler(optimizer_dis, 10000)

            results = {
                "losses": {
                    "train": [],
                    "valid": [],
                },
                "losses_gen": {
                    "train": [],
                    "valid": [],
                },
                "losses_dis_real": {
                    "train": [],
                    "valid": [],
                },
                "losses_dis_fake": {
                    "train": [],
                    "valid": [],
                }
            }

            early_stop_counter = 0
            print("\n\n\nCV:", cv, "/", self.params['cv'], "\nTrain samples:", len(train_set),
                  "\nValid samples:", len(valid_set), "\n\n\n")

            best_epoch = 0
            if epoch_offset >= self.params['n_epochs']:
                return best_dict['loss']
            for epoch in range(epoch_offset, self.params['n_epochs']):
                is_best_epoch = False
                traces = {
                    "losses": {
                        "train": [],
                        "valid": [],
                    },
                    "losses_gen": {
                        "train": [],
                        "valid": [],
                    },
                    "losses_dis_real": {
                        "train": [],
                        "valid": [],
                    },
                    "losses_dis_fake": {
                        "train": [],
                        "valid": [],
                    }
                }
                if early_stop_counter == self.early_stop:
                    if self.verbose > 0:
                        print('EARLY STOPPING.')
                    break

                gen.train()
                dis.train()
                fake_images = None
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
                        image = torch.Tensor(image.float()).to(device)
                        image = torch.cat((image, torch.zeros(image.shape[0], 1).to(device)), 1)

                        if self.model_name == 'dcgan':
                            noise = torch.rand([self.params['bs'], 64, 1]).to(device)
                            fake_images = gen(noise)
                        elif 'pacgan' in self.model_name or 'pacwgan' in self.model_name:
                            fake_images = []
                            for _ in range(self.n_pacgan):
                                noise = torch.rand([int(self.params['bs'] / self.n_pacgan), 64, 1]).to(device)
                                fake_images += [gen(noise).squeeze()]
                            fake_images = torch.stack(fake_images, 1)
                        elif self.model_name == 'gan':
                            noise = torch.rand([self.params['bs'], 100])
                            fake_images = gen(noise.to(device))
                        else:
                            exit('Not a valid model')
                        if j == 0:
                            plt.plot(fake_images.detach().cpu().numpy()[0][0])
                            plt.title(f"Epoch {epoch}")
                            plt.xlabel('m/z')
                            plt.ylabel('Normalized intensity')
                            os.makedirs(f"{self.params['checkpoint_path']}/spectra", exist_ok=True)
                            plt.savefig(f"{self.params['checkpoint_path']}/spectra/spectrum_train_1_{epoch}")
                            plt.close()

                            plt.plot(fake_images.detach().cpu().numpy()[1][0])
                            plt.title(f"Epoch {epoch}")
                            plt.xlabel('m/z')
                            plt.ylabel('Normalized intensity')
                            plt.savefig(
                                f"{self.params['checkpoint_path']}/{self.model_path}/spectra/spectrum_train_2_{epoch}")
                            plt.close()
                        ############################
                        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                        ###########################
                        # if j % 2 == 0:
                        dis.train()
                        dis.zero_grad()

                        if 'pacgan' in self.model_name or 'pacwgan' in self.model_name:
                            ims_list = []
                            assert len(image) % self.n_pacgan == 0
                            for i, im in enumerate(image):
                                if i % self.n_pacgan == 0:
                                    if i != 0:
                                        ims_list += [ims]
                                    ims = im.unsqueeze(0)
                                elif i % self.n_pacgan != 0:
                                    ims = torch.cat((ims, im.unsqueeze(0)), 0)
                            ims_list += [ims]
                            image = torch.stack(ims_list)
                            dis_preds = dis(image)
                        else:
                            dis_preds = dis(image.unsqueeze(1))

                        ones = torch.ones(dis_preds.shape[0]).to(device)
                        loss_dis_real = self.criterion(
                            dis_preds.squeeze(),
                            ones
                        ) / self.params['bs']
                        # if j % 2 == 0:
                        # loss_dis_real.backward()

                        zeros = torch.zeros(dis_preds.shape[0]).to(device)
                        dis_fake_preds = dis(fake_images.detach())
                        loss_dis_fake = self.criterion(
                            dis_fake_preds.squeeze(),
                            zeros
                        ) / self.params['bs']

                        # loss_dis = torch.Tensor([loss_dis_fake, loss_dis_real])
                        # loss_dis.requires_grad = True
                        err = torch.abs(loss_dis_real - loss_dis_fake)
                        err.backward()
                        # if j % 2 == 0:
                        # loss_dis_fake.backward()
                        optimizer_dis.step()

                        ############################
                        # (2) Update G network: maximize log(D(G(z)))
                        ###########################
                        # if j % 2 == 0:
                        # dis.eval()
                        for _ in range(3):
                            dis.zero_grad()
                            gen.zero_grad()
                            if self.model_name == 'dcgan':
                                noise = torch.rand([self.params['bs'], 64, 1]).to(device)
                                fake_images = gen(noise)
                            elif 'pacgan' in self.model_name or 'pacwgan' in self.model_name:
                                fake_images = []
                                for _ in range(self.n_pacgan):
                                    noise = torch.rand([int(self.params['bs'] / self.n_pacgan), 64, 1]).to(device)
                                    fake_images += [gen(noise).squeeze()]
                                fake_images = torch.stack(fake_images, 1)
                            elif self.model_name == 'gan':
                                noise = torch.rand([self.params['bs'], 100])
                                fake_images = gen(noise.to(device))
                            dis_fake_preds = dis(fake_images)
                            ones = torch.ones(fake_images.shape[0]).to(device)
                            loss_gen = self.criterion(
                                dis_fake_preds.squeeze(),  # .view(len(batch), -1),
                                ones
                            ) / self.params['bs']
                            # if j % 2 != 0:
                            loss_gen.backward()
                            optimizer_gen.step()

                        loss = loss_dis_real.item() + loss_dis_fake.item() + loss_gen.item()

                        traces['losses']['train'] += [loss]
                        traces['losses_gen']['train'] += [loss_gen.item()]
                        traces['losses_dis_real']['train'] += [loss_dis_real.item()]
                        traces['losses_dis_fake']['train'] += [loss_dis_fake.item()]

                        if self.params['scheduler'] == "CycleScheduler":
                            lr_scheduler_gen.step()
                            lr_scheduler_dis.step()
                            # optimizer = lr_scheduler.optimizer
                        del loss, loss_gen, loss_dis_fake, loss_dis_real, dis_preds, dis_fake_preds, ones, zeros, noise
                    del train_loader, train_set

                results["losses"]["train"] += [np.mean(traces['losses']['train'])]
                results["losses_gen"]["train"] += [np.mean(traces['losses_gen']['train'])]
                results["losses_dis_real"]["train"] += [np.mean(traces['losses_dis_real']['train'])]
                results["losses_dis_fake"]["train"] += [np.mean(traces['losses_dis_fake']['train'])]
                logger.add_scalar('train/loss', results["losses"]["train"][-1], epoch)
                logger.add_scalar('train/loss_gen', results["losses_gen"]["train"][-1], epoch)
                logger.add_scalar('train/loss_dis_real', results["losses_dis_real"]["train"][-1], epoch)
                logger.add_scalar('train/loss_dis_fake', results["losses_dis_fake"]["train"][-1], epoch)
                if epoch % self.epochs_per_print == 0:
                    if self.verbose > 1:
                        print_results(results, mode='train', epoch=epoch)

                gen.eval()
                dis.eval()
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
                            image = torch.cat((image, torch.zeros(image.shape[0], 1).to(device)), 1)

                            if 'pacgan' in self.model_name or 'pacwgan' in self.model_name:
                                ims_list = []
                                assert len(image) % self.n_pacgan == 0
                                for i, im in enumerate(image):
                                    if i % self.n_pacgan == 0:
                                        if i != 0:
                                            ims_list += [ims]
                                        ims = im.unsqueeze(0)
                                    elif i % self.n_pacgan != 0:
                                        ims = torch.cat((ims, im.unsqueeze(0)), 0)
                                ims_list += [ims]
                                image = torch.stack(ims_list)
                                dis_preds = dis(image)
                            else:
                                dis_preds = dis(image.unsqueeze(1))
                            if self.model_name == 'dcgan':
                                noise = torch.rand([self.params['bs'], 64, 1]).to(device)
                                fake_images = gen(noise)
                            elif 'pacgan' in self.model_name or 'pacwgan' in self.model_name:
                                fake_images = []
                                for _ in range(self.n_pacgan):
                                    noise = torch.rand([int(self.params['bs'] / self.n_pacgan), 64, 1]).to(device)
                                    fake_images += [gen(noise).squeeze()]
                                fake_images = torch.stack(fake_images, 1)
                            elif self.model_name == 'gan':
                                noise = torch.rand([self.params['bs'], 100]).to(device)
                                fake_images = gen(noise)
                            if j == 0:
                                plt.plot(fake_images.detach().cpu().numpy()[0][0])
                                plt.title(f"Epoch {epoch}")
                                plt.xlabel('m/z')
                                plt.ylabel('Normalized intensity')
                                os.makedirs(f"{self.params['checkpoint_path']}/spectra",
                                            exist_ok=True)
                                plt.savefig(
                                    f"{self.params['checkpoint_path']}/spectra/spectrum_valid_1_{epoch}")
                                plt.close()

                                plt.plot(fake_images.detach().cpu().numpy()[1][0])
                                plt.title(f"Epoch {epoch}")
                                plt.xlabel('m/z')
                                plt.ylabel('Normalized intensity')

                                plt.savefig(
                                    f"{self.params['checkpoint_path']}/spectra/spectrum_valid_2_{epoch}")
                                plt.close()

                            targets = torch.ones(dis_preds.shape[0]).to(device)
                            loss_dis_real = self.criterion(
                                dis_preds.squeeze(),  # .view(len(batch), -1),
                                targets
                            ) / self.params['bs']

                            dis_fake_preds = dis(fake_images)
                            targets = torch.zeros(dis_preds.shape[0]).to(device)
                            loss_dis_fake = self.criterion(
                                dis_fake_preds.squeeze(),  # .view(len(batch), -1),
                                targets
                            ) / self.params['bs']

                            targets = torch.ones(dis_preds.shape[0]).to(device)
                            loss_gen = self.criterion(
                                dis_fake_preds.squeeze(),  # .view(len(batch), -1),
                                targets
                            ) / self.params['bs']

                            loss = loss_dis_real + loss_dis_fake + loss_gen

                            traces['losses']['valid'] += [loss.item()]
                            traces['losses_gen']['valid'] += [loss_gen.item()]
                            traces['losses_dis_real']['valid'] += [loss_dis_real.item()]
                            traces['losses_dis_fake']['valid'] += [loss_dis_fake.item()]
                            # optimizer = lr_scheduler.optimizer
                            del loss, loss_gen, loss_dis_fake, loss_dis_real, dis_preds, dis_fake_preds, targets, noise
                        del valid_loader, valid_set
                results['losses']["valid"] += [np.mean(traces['losses']['valid'])]
                results['losses_gen']["valid"] += [np.mean(traces['losses_gen']['valid'])]
                results['losses_dis_real']["valid"] += [np.mean(traces['losses_dis_real']['valid'])]
                results['losses_dis_fake']["valid"] += [np.mean(traces['losses_dis_fake']['valid'])]
                logger.add_scalar('valid/loss', results['losses']['valid'][-1], epoch)
                logger.add_scalar('valid/losses_gen', results['losses_gen']['valid'][-1], epoch)
                logger.add_scalar('valid/losses_dis_real', results['losses_dis_real']['valid'][-1], epoch)
                logger.add_scalar('valid/losses_dis_fake', results['losses_dis_fake']['valid'][-1], epoch)
                if self.params['scheduler'] == "ReduceLROnPlateau":
                    if epoch - epoch_offset > 5:
                        lr_scheduler_gen.step(results['losses']["valid"][-1])
                        lr_scheduler_dis.step(results['losses']["valid"][-1])
                save_checkpoint(gen,
                                dis,
                                optimizer_gen,
                                optimizer_dis,
                                self.params['lr'],
                                epoch,
                                f"{self.params['checkpoint_path']}/{self.model_path}",
                                name=f"{self.model_name}"
                                )
                if (results['losses']['valid'][-1] < best_dict['loss'] or best_dict['loss'] == -1) and not np.isnan(
                        results['losses']['valid'][-1]):
                    if self.verbose > 1:
                        print('BEST EPOCH!', results['losses']['valid'][-1])
                    early_stop_counter = 0
                    best_dict['loss'] = results['losses']['valid'][-1]
                    best_dict['loss_dis_real'] = results['losses_dis_real']['valid'][-1]
                    best_dict['loss_dis_fake'] = results['losses_dis_fake']['valid'][-1]
                    best_dict['losses_gen'] = results['losses_gen']['valid'][-1]
                    is_best_epoch = True
                    best_epoch = epoch
                else:
                    early_stop_counter += 1

                if epoch % self.epochs_per_print == 0:
                    if self.verbose > 0:
                        print_results(results, mode='valid', epoch=epoch)

                    if self.verbose > 1:
                        print("Current LR:", optimizer_gen.param_groups[0]['lr'])
                    if 'momentum' in optimizer_gen.param_groups[0].keys():
                        print("Current Momentum:", optimizer_gen.param_groups[0]['momentum'])
            if self.verbose > 0:
                print('BEST LOSS :', best_dict['loss'])

        tb_logging.logging(results)

        return best_dict['loss']

    def get_scheduler(self, optimizer, n_samples=None):
        if self.params['scheduler'] == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                      factor=0.1,
                                                                      cooldown=0,
                                                                      patience=25,
                                                                      verbose=True,
                                                                      min_lr=1e-8)
        elif self.params['scheduler'] == 'CycleScheduler':
            assert n_samples is not None
            lr_scheduler = CycleScheduler(optimizer,
                                          self.params['lr'],
                                          n_iter=n_samples)
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
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--gen_out_activation", type=str, default='sigmoid')
    parser.add_argument("--dis_out_activation", type=str, default='linear')
    parser.add_argument("--model", type=str, default='pacwgan')
    parser.add_argument("--bs", type=int, default=8)
    args = parser.parse_args()

    if 'dcgan' in args.model:
        from src.models.pytorch.DCGAN_1D import Generator as gen
        from src.models.pytorch.DCGAN_1D import Discriminator as dis
    elif 'pacgan' in args.model or 'pacwgan' in args.model :
        from src.models.pytorch.PacGAN_1D_1 import Generator as gen
        from src.models.pytorch.PacGAN_1D_1 import Discriminator as dis
    elif "gan" in args.model:
        from src.models.pytorch.GAN_1D import Generator as gen
        from src.models.pytorch.GAN_1D import Discriminator as dis
    else:
        exit('WRONG MODEL NAME')

    params = {
        "max_value": np.nan,
        "cv": 5,
        "bn": True,
        "bs": args.bs,
        "n_epochs": 10000,
        "size": 256,
        'epochs_per_checkpoint': 1,
        'optimizer_type': 'adam',
        'scheduler': "ReduceLROnPlateau",
        "res_blocks": True,
        "has_dense": True,
        "checkpoint_path": args.checkpoint_path,
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
        gen,
        dis,
        model_name=args.model,
        gen_out_activation=args.gen_out_activation,
        dis_out_activation=args.dis_out_activation,
        images_path_train=args.images_path_train,
        images_path_valid=args.images_path_valid,
        targets_path_train=args.labels_path_train,
        targets_path_valid=args.labels_path_valid,
        loss=args.loss,
        checkpoint_path=args.checkpoint_path,
        save=True,
        load=True,
        early_stop=10000,
        save_checkpoints=False,
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
