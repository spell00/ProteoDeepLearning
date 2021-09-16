from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import numpy as np


class TensorboardLogging:

    def __init__(self, hparams_filepath, params):
        self.params = params
        self.hparams_filepath = hparams_filepath
        HP_EPOCHS = hp.HParam('epochs', hp.IntInterval(1, 50))
        HP_LR = hp.HParam('lr', hp.RealInterval(1e-6, 1e-3))
        HP_WD = hp.HParam('wd', hp.RealInterval(1e-8, 1e-3))
        HP_BS = hp.HParam('bs', hp.IntInterval(1, 512))
        HP_L1 = hp.HParam('l1', hp.IntInterval(1, 512))
        HPARAMS = [HP_EPOCHS, HP_LR, HP_WD, HP_BS, HP_L1]
        with tf.summary.create_file_writer(hparams_filepath).as_default():
            hp.hparams_config(
                hparams=HPARAMS,
                metrics=[
                    hp.Metric('train_accuracy', display_name='Train Accuracy'),
                    hp.Metric('valid_accuracy', display_name='Valid Accuracy'),
                    hp.Metric('test_accuracy', display_name='Test Accuracy'),
                    hp.Metric('train_loss', display_name='Train Loss'),
                    hp.Metric('valid_loss', display_name='Valid Loss'),
                    hp.Metric('test_loss', display_name='Test Loss'),
                    hp.Metric('train_mcc', display_name='Train MCC'),
                    hp.Metric('valid_mcc', display_name='Valid MCC'),
                    hp.Metric('test_mcc', display_name='Test MCC')
                ],
            )

    def logging(self, traces):
        epochs = self.params['n_epochs']
        lr = self.params['lr']
        wd = self.params['wd']
        l1 = self.params['l1']
        bs = self.params['bs']
        with tf.summary.create_file_writer(self.hparams_filepath).as_default():
            hp.hparams({
                'epochs': epochs,
                'lr': lr,
                'wd': wd,
                'bs': bs,
                'l1': l1,
            })  # record the values used in this trial
            tf.summary.scalar('train_accuracy', np.mean([np.mean(x) for x in traces['train']['accuracies']]), step=1)
            tf.summary.scalar('valid_accuracy', np.mean(traces['valid']['accuracies']), step=1)
            tf.summary.scalar('test_accuracy', np.mean(traces['test']['accuracies']), step=1)
            tf.summary.scalar('train_loss', np.mean([np.mean(x) for x in traces['train']['losses']]), step=1)
            tf.summary.scalar('valid_loss', np.mean(traces['valid']['losses']), step=1)
            tf.summary.scalar('test_loss', np.mean(traces['test']['losses']), step=1)
            tf.summary.scalar('train_mcc', np.mean([np.mean(x) for x in traces['train']['mccs']]), step=1)
            tf.summary.scalar('valid_mcc', np.mean(traces['valid']['mccs']), step=1)
            tf.summary.scalar('test_mcc', np.mean(traces['test']['mccs']), step=1)

