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
from src.utils.CycleAnnealScheduler import CycleScheduler
from src.utils.dataset import MSDataset, MetaMSDataset
from src.models.pytorch.utils.checkpoints import load_checkpoint, save_checkpoint
import torch.nn.functional as F
from src.utils.logging import TensorboardLogging
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


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


# Find a gaussian for both outputs
def kde(gen, dis, loader, device='cuda'):
    from scipy import stats
    dis.eval()
    gen.eval()
    ps = []
    qs = []
    for sample in loader:
        sample = sample.to(device)
        noise = torch.rand([loader.batch_size, 64, 1]).to(device)
        real = dis(sample.unsqueeze(1)).squeeze()
        real_outs = torch.softmax(real, 1)
        p = [out.max().item() for out in real_outs]
        fake = gen(noise)
        fake_outs = dis(fake).squeeze()
        fake_outs = torch.softmax(fake_outs, 1)
        q = [out.max().item() for out in fake_outs]

        ps.extend(p)
        qs.extend(q)
    gkde = stats.gaussian_kde(ps)
    estimated_pdf = gkde.evaluate(qs)
    return estimated_pdf


def get_traces():
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
    return traces


def get_results_empty():
    results_arr = {
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

    return results_arr


class Train:
    def __init__(self,
                 params,
                 gen, dis,
                 mss_path_train,
                 mss_path_valid,
                 loss,
                 checkpoint_path,
                 model_name,
                 gen_out_activation,
                 dis_out_activation,
                 n_gen,
                 n_dis,
                 fp16_run=False,
                 epochs_per_checkpoint=1,
                 epochs_per_print=1,
                 verbose=2,
                 early_stop=100,
                 load=True,
                 save_checkpoints=True,
                 save=True,
                 norm='bn',
                 bin='1',
                 log1p=False,
                 smoothing_value=0.,
                 clip_value=0,
                 ):
        super().__init__()
        self.n_gen = n_gen
        self.n_dis = n_dis
        self.log1p = log1p
        self.clip_value = clip_value
        if clip_value == 0:
            self.clip = False
        else:
            self.clip = True

        self.smoothing_value = smoothing_value
        self.clip_value = clip_value

        self.gen_out_activation = gen_out_activation
        self.dis_out_activation = dis_out_activation
        self.fp16_run = fp16_run
        self.bin = bin
        self.norm = norm
        self.model_name = f"{model_name}"
        if 'pacgan' in self.model_name:
            self.n_pacgan = int(self.model_name.split('pacgan')[1])
        elif 'pacwgan' in self.model_name:
            self.n_pacgan = int(self.model_name.split('pacwgan')[1])
        else:
            self.n_pacgan = None
        if gen_out_activation == 'sigmoid':
            self.gen = gen(
                out_activation=nn.Sigmoid,
                norm=norm
            ).to(device)
        elif gen_out_activation == 'linear':
            self.gen = gen(
                out_activation=None,
                norm=norm
            ).to(device)
        if dis_out_activation == 'sigmoid':
            self.dis = dis(
                out_activation=nn.Sigmoid,
                in_channel=self.n_pacgan,
                norm=norm
            ).to(device)
        elif dis_out_activation == 'hardsigmoid':
            self.dis = dis(
                out_activation=nn.Hardsigmoid,
                in_channel=self.n_pacgan,
                norm=norm
            ).to(device)
        elif dis_out_activation == 'linear':
            self.dis = dis(
                out_activation=None,
                in_channel=self.n_pacgan,
                norm=norm
            ).to(device)

        self.checkpoint_path = checkpoint_path
        self.epochs_per_checkpoint = epochs_per_checkpoint
        self.epochs_per_print = epochs_per_print
        self.save = save
        if loss == 'bce':
            self.criterion_g = nn.BCEWithLogitsLoss()
            self.criterion_d = nn.BCEWithLogitsLoss()
        elif loss == 'mse':
            self.criterion_g = nn.MSELoss()
            self.criterion_d = nn.MSELoss()
        else:
            exit(f'{loss} is not implemented. Valid choices are: [bce, dice, jaccard]')

        self.save_checkpoints = save_checkpoints
        self.load = load
        self.loss_type = loss
        self.verbose = verbose
        self.mss_path_train = mss_path_train
        self.mss_path_valid = mss_path_valid
        self.early_stop = early_stop
        self.params = params

    def get_fake_tensor(self, grad=False):
        if grad:
            fake_mss = self._get_fake_tensor()
        else:
            with torch.no_grad():
                fake_mss = self._get_fake_tensor()

        return fake_mss

    def _get_fake_tensor(self):
        if self.model_name == 'dcgan':
            noise = torch.rand([self.params['bs'], 64, 1]).to(device)
            fake_mss = self.gen(noise)
        elif 'pacgan' in self.model_name or 'pacwgan' in self.model_name:
            fake_mss = []
            for _ in range(self.n_pacgan):
                noise = torch.rand([int(self.params['bs'] / self.n_pacgan), 64, 1]).to(device)
                fake_mss += [self.gen(noise).squeeze()]
            fake_mss = torch.stack(fake_mss, 1)
        elif self.model_name == 'gan':
            noise = torch.rand([self.params['bs'], 100])
            fake_mss = self.gen(noise.to(device))
        return fake_mss

    def get_real_tensor(self, spectrum):
        if 'pacgan' in self.model_name or 'pacwgan' in self.model_name:
            ims_list = []
            assert len(spectrum) % self.n_pacgan == 0
            for i, im in enumerate(spectrum):
                if i % self.n_pacgan == 0:
                    if i != 0:
                        ims_list += [ims]
                    ims = im.unsqueeze(0)
                elif i % self.n_pacgan != 0:
                    ims = torch.cat((ims, im.unsqueeze(0)), 0)
            ims_list += [ims]
            ms = torch.stack(ims_list)
            dis_preds = self.dis(ms)
        else:
            dis_preds = self.dis(spectrum.unsqueeze(1))
        return dis_preds

    def plot_ms(self, fake_mss, epoch, set='train'):
        plt.plot(fake_mss.detach().cpu().numpy()[0][0])
        plt.title(f"Epoch {epoch}")
        plt.xlabel('m/z')
        plt.ylabel('Normalized intensity')
        os.makedirs(f"{self.params['checkpoint_path']}/{self.model_path}/spectra",
                    exist_ok=True)
        try:
            plt.savefig(
                f"{self.params['checkpoint_path']}/{self.model_path}/spectra/spectrum_{set}_1_{epoch}")
        except:
            print('error saving ms')
        plt.close()

        plt.plot(fake_mss.detach().cpu().numpy()[1][0])
        plt.title(f"Epoch {epoch}")
        plt.xlabel('m/z')
        plt.ylabel('Normalized intensity')

        try:
            plt.savefig(
                f"{self.params['checkpoint_path']}/{self.model_path}/spectra/spectrum_{set}_2_{epoch}")
        except:
            print('error saving ms')
        plt.close()

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

    def fid(self):
        # TODO Frechet Inception Distance
        pass

    def spectral_norm(self):
        # TODO Implement Spectral Normalization (NOT HERE)
        pass

    def multiscale(self):
        # TODO Implement multiscale GAN (NOT HERE)
        pass

    def train(self, params_dict):
        global tb_logging, results
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
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
        self.optimizer_gen = get_optimizer(self.gen, self.params)
        self.optimizer_dis = get_optimizer(self.dis, self.params)

        wd_str = str(self.params["wd"])
        l1_str = str(self.params["l1"])
        lr_str = str(self.params["lr"])
        wd_str = wd_str[:5] + wd_str[-4:]
        l1_str = l1_str[:5] + l1_str[-4:]
        lr_str = lr_str[:5] + lr_str[-4:]
        scale_str = str(np.round(self.params['scale'], 2))
        drop_str = str(np.round(self.params['dropout'], 1))
        self.model_path = '/'.join([
            f"bin{self.bin}",
            f"gen{self.gen_out_activation}",
            f"dis{self.dis_out_activation}",
            self.params['scheduler'],
            f'{self.loss_type}',
            f'{self.norm}',
            f'log{self.log1p}',
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

        print("\nGenerator total parameters\n")
        self.gen.count_parameters()
        print("\nDiscriminator total parameters\n")
        self.dis.count_parameters()
        train_set = MetaMSDataset(self.mss_path_train, load_all=False)
        valid_set = MetaMSDataset(self.mss_path_valid, load_all=False)
        # spliter = validation_spliter(all_set, cv=self.params['cv'])

        # TODO Make a test set

        # TODO replace with no CV (or cv=1 will be no CV?)
        epoch_offset = 0
        if self.load:
            self.gen, dis, optimizer_gen, optimizer_dis, epoch_offset = load_checkpoint(
                f"{self.params['checkpoint_path']}/{self.model_path}",
                self.gen,
                self.dis,
                self.optimizer_gen,
                self.optimizer_dis,
                self.model_name
            )

        self.optimizer_gen.param_groups[0]['lr'] = self.params['lr'] / 10
        self.optimizer_dis.param_groups[0]['lr'] = self.params['lr']

        epoch_offset = max(1, epoch_offset + 1)
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

        lr_scheduler_gen = self.get_scheduler(self.optimizer_gen, 10000)
        lr_scheduler_dis = self.get_scheduler(self.optimizer_dis, 10000)
        results = get_results_empty()
        early_stop_counter = 0

        # means = torch.load('stats/torch/all_means')
        # stds = torch.load('stats/torch/all_stds')

        print("\n\n\nTrain samples:", len(train_set), "\nValid samples:", len(valid_set), "\n\n\n")

        best_epoch = 0
        if epoch_offset >= self.params['n_epochs']:
            return best_dict['loss']
        for epoch in range(epoch_offset, self.params['n_epochs']):
            is_best_epoch = False
            traces = get_traces()
            if early_stop_counter == self.early_stop:
                if self.verbose > 0:
                    print('EARLY STOPPING.')
                break
            self.gen.train()
            self.dis.train()
            # pbar = tqdm(total=len(train_loader))
            for i, (mss, _, columns) in enumerate(meta_train_loader):
                columns = [c[0] for c in columns]
                train_set = MSDataset(mss.squeeze(), columns, log=self.log1p, means=None, stds=None)

                train_loader = DataLoader(train_set,
                                          num_workers=0,
                                          shuffle=True,
                                          batch_size=self.params['bs'],
                                          pin_memory=True,
                                          drop_last=True)
                for j, (ms, pmz, tics, rtime) in enumerate(train_loader):
                    # pbar.update(1)
                    ms = torch.Tensor(ms.float()).to(device)
                    fake_mss = self.get_fake_tensor()
                    if j == 0:
                        if self.save:
                            self.plot_ms(fake_mss, epoch, 'train')
                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    self.dis.train()
                    self.dis.zero_grad()
                    dis_preds = self.get_real_tensor(ms)

                    # ones = torch.FloatTensor([1]).to(device)
                    # mones = ones * -1

                    # if self.loss_type != 'bce':
                    #     dis_preds = torch.sigmoid(dis_preds)

                    ones = torch.ones(fake_mss.shape[0]).to(device)
                    loss_dis_real = self.criterion_d(
                        dis_preds.squeeze(),
                        ones
                    )  # / self.params['bs']

                    # loss_dis_real = loss_dis_real.mean(0).view(1)
                    loss_dis_real.backward()

                    mones = -torch.ones(dis_preds.shape[0]).to(device) + self.smoothing_value
                    dis_fake_preds = self.dis(fake_mss.detach())
                    # if self.loss_type != 'bce':
                    #     dis_fake_preds = torch.sigmoid(dis_fake_preds)
                    loss_dis_fake = self.criterion_d(
                        dis_fake_preds.squeeze(),
                        mones
                    )  # / self.params['bs']
                    # loss_dis_fake = loss_dis_fake.mean(0).view(1)
                    loss_dis_fake.backward()
                    self.optimizer_dis.step()

                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################
                    # Clip weights of discriminator
                    if self.clip:
                        for p in self.dis.parameters():
                            p.data.clamp_(-self.clip_value, self.clip_value)

                    if j % self.n_dis == 0:
                        loss_gen = self.train_gen(n=self.n_gen)
                        loss = loss_dis_real.item() + loss_dis_fake.item() + loss_gen.item()
                        traces['losses']['train'] += [loss]
                        traces['losses_gen']['train'] += [loss_gen.item()]
                        traces['losses_dis_real']['train'] += [loss_dis_real.item()]
                        traces['losses_dis_fake']['train'] += [loss_dis_fake.item()]
                        del loss, loss_gen

                    if self.params['scheduler'] == "CycleScheduler":
                        lr_scheduler_gen.step()
                        lr_scheduler_dis.step()
                        # optimizer = lr_schedu     ler.optimizer
                    del loss_dis_fake, loss_dis_real, dis_preds, dis_fake_preds
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

            self.gen.eval()
            self.dis.eval()
            with torch.no_grad():
                for i, (mss, _, columns) in enumerate(meta_valid_loader):
                    valid_set = MSDataset(mss, columns, log=self.log1p, means=None, stds=None)

                    valid_loader = DataLoader(valid_set,
                                              num_workers=0,
                                              shuffle=True,
                                              batch_size=self.params['bs'],
                                              pin_memory=True,
                                              drop_last=True)
                    for j, (ms, pmz, tics, rtime) in enumerate(valid_loader):
                        ms = torch.Tensor(ms.float()).to(device)

                        dis_preds = self.get_real_tensor(ms)
                        # loss_dis_real = loss_dis_real.mean(0).view(1)
                        fake_mss = self.get_fake_tensor()
                        if j == 0:
                            if self.save:
                                self.plot_ms(fake_mss, epoch, 'valid')

                        ones = torch.ones(dis_preds.shape[0]).to(device)
                        # if self.loss_type != 'bce':
                        #     dis_preds = torch.sigmoid(dis_preds)
                        loss_dis_real = self.criterion_d(
                            dis_preds.squeeze(),  # .view(len(batch), -1),
                            ones
                        )  # / self.params['bs']

                        dis_fake_preds = self.dis(fake_mss)
                        # loss_dis_fake = loss_dis_fake.mean(0).view(1)
                        # if self.loss_type != 'bce':
                        #    dis_fake_preds = torch.sigmoid(dis_fake_preds)
                        mones = -torch.ones(dis_preds.shape[0]).to(device)
                        loss_dis_fake = self.criterion_d(
                            dis_fake_preds.squeeze(),  # .view(len(batch), -1),
                            mones
                        )  # / self.params['bs']

                        ones = torch.ones(dis_preds.shape[0]).to(device)
                        loss_gen = self.criterion_g(
                            dis_fake_preds.squeeze(),  # .view(len(batch), -1),
                            ones
                        )  # / self.params['bs']

                        loss = loss_dis_real + loss_dis_fake + loss_gen

                        traces['losses']['valid'] += [loss.item()]
                        traces['losses_gen']['valid'] += [loss_gen.item()]
                        traces['losses_dis_real']['valid'] += [loss_dis_real.item()]
                        traces['losses_dis_fake']['valid'] += [loss_dis_fake.item()]
                        # optimizer = lr_scheduler.optimizer
                        del loss, loss_gen, loss_dis_fake, loss_dis_real, dis_preds, dis_fake_preds
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
            if self.save:
                save_checkpoint(self.gen,
                                self.dis,
                                self.optimizer_gen,
                                self.optimizer_dis,
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
                    print("Current LR:", self.optimizer_gen.param_groups[0]['lr'])
                if 'momentum' in self.optimizer_gen.param_groups[0].keys():
                    print("Current Momentum:", self.optimizer_gen.param_groups[0]['momentum'])
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

    def get_volume(self, ms):
        self.dis.zero_grad()
        self.gen.zero_grad()
        fake_mss = self.get_fake_tensor()
        dis_fake_preds = self.dis(fake_mss)
        ones = torch.ones(fake_mss.shape[0]).to(device)
        loss_gen = self.criterion_g(
            dis_fake_preds.squeeze(),
            ones
        )  # / self.params['bs']

        loss_gen.backward()
        self.optimizer_gen.step()

        return loss_gen

    def train_gen(self, n):
        for _ in range(n):
            self.dis.zero_grad()
            self.gen.zero_grad()
            fake_mss = self.get_fake_tensor(grad=True)
            dis_fake_preds = self.dis(fake_mss)
            ones = torch.ones(fake_mss.shape[0]).to(device)
            loss_gen = self.criterion_g(
                dis_fake_preds.squeeze(),
                ones
            )  # / self.params['bs']
            # loss_gen = loss_gen.mean(0).view(1)
            loss_gen.backward()
            self.optimizer_gen.step()

        return loss_gen

    def train_dis(self, n):
        pass


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--mss_path_train", type=str, help="Path to intensities csv file",
                        default='inputs_bin1/g30/m2/train_numpy')
    parser.add_argument("--mss_path_valid", type=str, help="Path to intensities csv file",
                        default='inputs_bin1/g30/m2/valid_numpy')
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Checkpoint path")
    parser.add_argument("--loss", type=str, default="mse", help="Path to labels csv file")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--gen_out_activation", type=str, default='sigmoid')
    parser.add_argument("--dis_out_activation", type=str, default='linear')
    parser.add_argument("--model", type=str, default='pacgan')
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--bin", type=str, default='1')
    parser.add_argument("--log1p", type=int, default=0)
    parser.add_argument("--norm", type=str, default='sn')
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

    if 'dcgan' in args.model:
        from src.models.pytorch.unsupervised.GANs.DCGAN_1D import Generator as gen
        from src.models.pytorch.unsupervised.GANs.DCGAN_1D import Discriminator as dis
    elif 'pacgan' in args.model or 'pacwgan' in args.model:
        from src.models.pytorch.unsupervised.GANs.PacGAN_1D1 import Generator as gen
        from src.models.pytorch.unsupervised.GANs.PacGAN_1D1 import Discriminator as dis
    elif "gan" in args.model:
        from src.models.pytorch.unsupervised.GANs.GAN_1D import Generator as gen
        from src.models.pytorch.unsupervised.GANs.GAN_1D import Discriminator as dis
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
        n_gen=args.n_gen,
        n_dis=args.n_dis,
        model_name=args.model,
        gen_out_activation=args.gen_out_activation,
        dis_out_activation=args.dis_out_activation,
        mss_path_train=args.mss_path_train,
        mss_path_valid=args.mss_path_valid,
        loss=args.loss,
        checkpoint_path=args.checkpoint_path,
        save=True,
        load=args.load,
        early_stop=10000,
        save_checkpoints=False,
        bin=args.bin,
        norm=args.norm,
        log1p=args.log1p,
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
            {"name": "learning_rate", "type": "range", "bounds": [1e-5, 1e-4], "log_scale": True},
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
