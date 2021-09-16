#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import os
import json
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import random
from tensorflow.keras.regularizers import l2

from src.models.keras.CNN import CNN
from src.models.keras.Linear import Linear
from src.utils.utils import getScalerFromString
from src.utils.metrics import matthews_correlation as mcc
from src.utils.logging import TensorboardLogging

from numpy.random import seed
import tensorflow as tf

from skopt.space import Real, Integer
from skopt import gp_minimize
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

# os.makedirs(f'logs/hparam_tuning', exist_ok=True)


def compute_confusion_matrix(y_test, y_classes):
    tn, fp, fn, tp = confusion_matrix(y_test, y_classes).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return sensitivity, specificity



def split_train_test(labels):
    from sklearn.model_selection import StratifiedKFold
    # First, get all unique samples and their category
    unique_samples = np.arange(0, len(labels))
    unique_cats = []
    # StratifiedKFold with n_splits of 5 to ranmdomly split 80/20.
    # Used only once for train/test split.
    # The train split needs to be split again into train/valid sets later
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    train_inds, test_inds = next(skf.split(np.zeros_like(labels), labels))

    # After the samples are split, we get the duplicates of all samples.
    # train_samples, test_samples = [unique_samples[s] for s in train_inds], [unique_samples[s] for s in test_inds]
    train_cats = [labels[ind] for ind in train_inds]

    assert len(unique_samples) == len(train_inds) + len(test_inds)
    assert len([x for x in test_inds if x in train_inds]) == 0
    # assert len([x for x in test_samples if x in train_samples]) == 0
    assert len(np.unique([labels[ind] for ind in test_inds])) > 1

    return train_inds, test_inds, train_cats


def ms_data(fname, minmax=False, flattening=False):
    from sklearn.preprocessing import minmax_scale
    mat_data = pd.read_csv(fname)
    samples = mat_data.index.values
    labels = np.array([sample.split('-')[0] for sample in np.unique(samples)])
    categories = np.array([int(lab.split('-')[1]) - 1 for lab in np.unique(samples)])
    samples_list = []
    for sample in np.unique(samples):
        samples_list += [mat_data.loc[sample].values]
        if flattening:
            samples_list[-1] = samples_list[-1].flatten()
    mat_data = np.stack(samples_list)
    # assert len(mat_data) == len(samples) == len(categories)
    # TODO Make sure this can be removed without =bad consequences. Not supposed to be here, but maybe it is good too
    if minmax:
        mat_data = minmax_scale(mat_data, axis=0, feature_range=(0, 1))
    mat_data = mat_data.astype("float32")
    return mat_data, labels, categories


class Train:
    def __init__(
            self,
            intensities_file,
            scaler,
            criterion='categorical_crossentropy',
            variant='logistic',
            get_data_function=ms_data,
            n_channels=1,
            save_train_models=False,
            model_name=CNN,
            verbose=0,
            model_path='None',
            freeze=True,
            retrain=False,
    ):
        self.dataset_name = intensities_file.split('/')[-1].split('_')[0]
        self.freeze = freeze
        self.verbose = verbose
        self.variant = variant
        self.scaler = scaler
        self.retrain = retrain
        self.model_name = model_name

        # If running for a final model, should put to True. Otherwise, in this case it makes it run more than 10x longer
        self.save_train_models = save_train_models
        self.data, labels_str, samples = ms_data(intensities_file)
        if 'Blk' in labels_str:
            for i, label in enumerate(labels_str):
                if label != 'Blk':
                    labels_str[i] = 'Not Blk'
        unique_labels = np.unique(labels_str)
        labels = np.array([np.argwhere(lab == unique_labels)[0][0] for lab in labels_str])
        self.data[np.isnan(self.data)] = 0
        self.best_acc = 0

        # TODO have this step done in R import_data.R

        categories = pd.Categorical(labels).codes
        self.labels_df = pd.concat([
            pd.DataFrame(np.array(samples).reshape([-1, 1])),
            pd.DataFrame(np.array(categories).reshape([-1, 1])),
            pd.DataFrame(np.array(labels).reshape([-1, 1])),
        ], 1)
        self.labels_df.columns = ['sample', 'category', 'label']
        self.nb_classes = len(np.unique(self.labels_df['category']))
        self.input_shape = [self.data.shape[1], n_channels]
        self.criterion = criterion
        self.step = 0
        self.call_num = 0
        self.previous_datasets = ""
        if not (model_path != 'None' or model_path is not None):
            self.model_path, self.previous_datasets = self.select_best_model(model_path)
        else:
            self.model_path = False
        if self.previous_datasets == "":
            self.datasets = self.dataset_name
        else:
            self.datasets = f"{self.previous_datasets}_{self.dataset_name}"
        self.params = {}

    def select_best_model(self, params_fname):
        # import from json file
        params = pd.read_json(params_fname)
        best_call = 0
        best_valid_loss = np.inf
        for call in list(params.keys()):
            values = params[call]
            if float(values['scores']['valid_loss']) < best_valid_loss:
                best_valid_loss = float(values['scores']['valid_loss'])
                best_call = call
        hparams = params[best_call]['h_params']
        base_path = params_fname.split('/')
        previous_datasets = params[best_call]['datasets']
        assert previous_datasets == base_path[4]
        base_path = f"{base_path[0]}/{base_path[1]}/{base_path[2]}/{base_path[3]}/{base_path[4]}/{base_path[5]}/{base_path[6]}"

        path = \
            f'{self.criterion}/' + \
            f'{"{:.8f}".format(float(hparams["lr"]))}/' \
            f'{"{:.8f}".format(float(hparams["wd"]))}/' \
            f'{"{:.8f}".format(float(hparams["l1"]))}/' \
            f'{hparams["bs"]}/'

        dir_path = f"{base_path}/{path}"

        return dir_path, previous_datasets

    def update_model(self, model_source, path, wd):
        model_source.model.build([1, 6150])
        model_source.model.load_weights(f"{path}/{self.variant}.h5")

        for layer in model_source.model.layers:
            layer.trainable = self.freeze
        if self.variant == 'lecun':
            model_source.model.pop()
            model_source.model.pop()

            new_model = Sequential([
                Dense(2050, activation='relu', activity_regularizer=l2(wd)),
                Dropout(0.5),
                Dense(self.nb_classes, activation='softmax', activity_regularizer=l2(wd))
            ])

            for layer in new_model.layers:
                layer.trainable = self.freeze

            model_source.model = Sequential([
                model_source.model,
                new_model
            ])
        elif self.variant == 'lenet':
            model_source.model.pop()
            model_source.model.pop()
            model_source.model.pop()
            model_source.model = Sequential([
                model_source,
                Flatten(),
                Dense(120, activation='relu', activity_regularizer=l2(wd)),
                Dense(84, activity_regularizer=l2(wd)),
                Dense(self.nb_classes, activation='softmax', activity_regularizer=l2(wd))
            ])
        elif self.variant == 'vgg9':
            model_source.model.pop()
            model_source.model.pop()
            model_source.model.pop()
            model_source.model = Sequential([
                model_source.model,
                Flatten(),
                Dense(4096, activation='relu', activity_regularizer=l2(wd)),
                Dropout(0.5),
                Dense(4096, activation='relu', activity_regularizer=l2(wd)),
                Dropout(0.5),
                Dense(self.nb_classes, activation='softmax', activity_regularizer=l2(wd))
            ])

        elif self.variant == 'logistic':
            pass
        else:
            exit("Wrong variant. Chose lenet, lecun, vgg9 or logistic ")
        return model_source

    def train(self, h_params):
        self.call_num += 1
        # n_epochs = h_params[0]
        lr = h_params[0]
        wd = h_params[1]
        l1 = h_params[2]
        bs = h_params[3]

        h_params = {
            # "n_epochs": n_epochs,
            "lr": lr,
            "wd": wd,
            "l1": l1,
            "bs": bs,
        }

        path = \
            f'{self.criterion}/' + \
            f'{"{:.8f}".format(float(lr))}/' \
            f'{"{:.8f}".format(float(wd))}/' \
            f'{"{:.8f}".format(float(l1))}/' \
            f'{bs}/'
        dnn = self.model_name(h_params, self.nb_classes, variant=self.variant, activation='relu', batch_size=bs)
        hparams_filepath = f"logs/{dnn.get_model_name()}/{self.variant}/{self.datasets}/{self.freeze}/{self.retrain}" \
                           f"/hparam_tuning/{path}"
        log_filepath = f"logs/{dnn.get_model_name()}/{self.variant}/{self.datasets}/{self.freeze}/{self.retrain}/{path}"
        # dnn.model.summary()
        # del dnn
        os.makedirs(log_filepath, exist_ok=True)
        tb_logging = TensorboardLogging(hparams_filepath, h_params)
        traces = {
            "train": {
                "losses": [],
                "accuracies": [],
                "mccs": [],
                "sensitivities": [],
                "specificities": [],
            },
            "valid": {
                "losses": [],
                "accuracies": [],
                "mccs": [],
                "sensitivities": [],
                "specificities": [],
            },
            "test": {
                "losses": [],
                "accuracies": [],
                "mccs": [],
                "sensitivities": [],
                "specificities": [],
            },
        }
        all_train_indices, test_indices, _ = split_train_test(self.labels_df['category'])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        all_x_train = self.data[all_train_indices]
        all_y_train = self.labels_df['category'][all_train_indices].values
        x_test = self.data[test_indices]
        y_test = self.labels_df['category'][test_indices].values

        assert len(set(y_test)) == self.nb_classes

        # 3-fold CV; there is only 8 Normal samples for canis sarcoma, so will avoid having only 1 Normal per set
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        for i, (train_inds, valid_inds) in enumerate(skf.split(np.zeros_like(all_y_train), all_y_train)):
            # Just plot the first iteration, it will already be crowded if doing > 100 optimization iterations
            if self.verbose:
                print(f"CV: {i + 1}/{5}")
            # new_nums = [train_nums[i] for i in inds]

            x_train = all_x_train[train_inds]
            y_train = all_y_train[train_inds]
            x_valid = all_x_train[valid_inds]
            y_valid = all_y_train[valid_inds]

            shape = list(x_train.shape[1:])

            assert len(set(y_train)) == self.nb_classes and len(set(y_valid)) == self.nb_classes
            assert len(all_train_indices) == len(train_inds) + len(valid_inds)
            assert len([x for x in valid_inds if x in train_inds]) == 0

            scaler = getScalerFromString(self.scaler)()
            x_train = x_train.reshape([x_train.shape[0], np.prod(shape)])
            x_valid = x_valid.reshape([x_valid.shape[0], np.prod(shape)])
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_valid = scaler.transform(x_valid)

            x_train_conv = np.reshape(x_train, [x_train.shape[0]] + shape)
            x_valid_conv = np.reshape(x_valid, [x_valid.shape[0]] + shape)
            y_train_conv = to_categorical(y_train, self.nb_classes)
            y_valid_conv = to_categorical(y_valid, self.nb_classes)

            dnn = self.model_name(h_params, self.nb_classes, batch_size=bs, variant=self.variant, activation='relu')
            dnn.build(shape)

            if self.model_path:
                dnn = self.update_model(model_source=dnn, path=self.model_path, wd=wd)
            dnn.model.compile(loss=self.criterion,
                              optimizer='adam',
                              metrics=['accuracy', mcc])
            file_path = f'saved_models/keras/{dnn.get_model_name()}/{self.variant}/{self.datasets}/checkpoint/'  # \
            os.makedirs(file_path, exist_ok=True)
            callbacks = []
            callbacks += [keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.1,
                patience=10,
                verbose=1,
                min_lr=1e-14)
            ]
            if i == 0:
                callbacks += [keras.callbacks.TensorBoard(
                    log_dir=log_filepath,
                    histogram_freq=1,
                    write_graph=True,
                    write_images=False,
                    update_freq="epoch",
                    profile_batch=2,
                    embeddings_freq=0,
                    embeddings_metadata=None,
                )]
            if self.save_train_models:
                callbacks += [tf.keras.callbacks.ModelCheckpoint(
                    file_path,
                    monitor="val_loss",
                    verbose=self.verbose,
                    save_best_only=True,
                    save_weights_only=False,
                    mode="auto",
                    save_freq="epoch",
                    options=None,
                )]
            callbacks += [keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=30,
                verbose=self.verbose,
                mode='min'
            )]

            y_integers = np.argmax(y_train_conv, axis=1)
            class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
            d_class_weights = dict(enumerate(class_weights))

            # CAN't USE VALIDATION SPLIT, so also can't use early_stopping or reduceLROnPLateau
            if self.verbose == 2:
                fit_verbose = 1
            else:
                fit_verbose = 0
            history = dnn.model.fit(
                x=x_train_conv,
                y=y_train_conv,
                batch_size=bs,
                verbose=fit_verbose,
                epochs=1000,
                # validation_split=0.,
                class_weight=d_class_weights,
                callbacks=callbacks,
                validation_data=(x_valid_conv, y_valid_conv)
            )
            # del dnn.model
            if self.save_train_models:
                dnn.model = tf.keras.models.load_model(f'{file_path}/checkpoint', custom_objects={
                    'mcc': mcc,
                })
                dnn.model.compile(loss=self.criterion,
                                  optimizer='adam',
                                  metrics=['accuracy', mcc])

            train_acc = history.history['accuracy']
            train_loss = history.history['loss']
            train_mcc = history.history['matthews_correlation']

            # train_loss, train_acc, train_mcc = dnn.model.evaluate(x_train_conv, y_train_conv, verbose=self.verbose)
            y_classes_train = np.argmax(dnn.model.predict(x_train_conv), axis=-1)
            # train_sensitivity, train_specificity = compute_confusion_matrix(y_train, y_classes_train)

            valid_loss, valid_acc, valid_mcc = dnn.model.evaluate(x_valid_conv, y_valid_conv, verbose=self.verbose)
            y_classes_valid = np.argmax(dnn.model.predict(x_valid_conv), axis=-1)
            # valid_sensitivity, valid_specificity = compute_confusion_matrix(y_valid, y_classes_valid)

            best_epoch = np.argmin(valid_loss)

            traces['train']['losses'].append(train_loss)
            traces['train']['accuracies'].append(train_acc)
            traces['train']['mccs'].append(train_mcc)
            # traces['train']['sensitivities'].append(train_sensitivity)
            # traces['train']['specificities'].append(train_specificity)
            traces['valid']['losses'].append(valid_loss)
            traces['valid']['accuracies'].append(valid_acc)
            traces['valid']['mccs'].append(valid_mcc)
            # traces['valid']['sensitivities'].append(valid_sensitivity)
            # traces['valid']['specificities'].append(valid_specificity)

        if self.retrain:
            x_all_train = self.data[all_train_indices]
            y_all_train = self.labels_df['category'][all_train_indices]

            scaler = getScalerFromString('robust')()
            scaler.fit(x_all_train)
            all_x_train = scaler.transform(x_all_train)

            x_train_conv = np.reshape(all_x_train, (all_x_train.shape[0], all_x_train.shape[1], 1))
            y_train_conv = to_categorical(y_all_train, self.nb_classes)

            x_test_trans = scaler.transform(x_test)
            x_test_conv = np.reshape(x_test_trans, (x_test_trans.shape[0], x_test_trans.shape[1], 1))
            y_test_conv = to_categorical(y_test, self.nb_classes)
            history = dnn.model.fit(
                x=x_train_conv,
                y=y_train_conv,
                batch_size=bs,
                verbose=fit_verbose,
                epochs=1000,
                class_weight=d_class_weights,
                callbacks=callbacks,
                validation_data=(x_test_conv, y_test_conv)
            )

        x_test = x_test.reshape([x_test.shape[0], np.prod(shape)])
        x_test = scaler.transform(x_test)
        x_test_conv = np.reshape(x_test, [x_test.shape[0]] + shape)
        y_test_conv = to_categorical(y_test, self.nb_classes)

        test_loss, test_acc, test_mcc = dnn.model.evaluate(x_test_conv, y_test_conv, verbose=self.verbose)
        y_classes_test = np.argmax(dnn.model.predict(x_test_conv), axis=-1)
        # test_sensitivity, test_specificity = compute_confusion_matrix(y_test, y_classes_test)
        traces['test']['losses'].append(test_loss)
        traces['test']['accuracies'].append(test_acc)
        traces['test']['mccs'].append(test_mcc)
        # traces['test']['specificities'].append(test_specificity)
        # traces['test']['sensitivities'].append(test_sensitivity)
        file_path = f'saved_models/keras/{dnn.get_model_name()}/{self.variant}/{self.datasets}/checkpoint/'  # \

        if test_acc > self.best_acc:
            print('Best model yet, saving...')
            self.best_acc = test_acc
            dnn.model.save_weights(f'{file_path}/{self.variant}.h5')
        self.step += 1
        try:
            tb_logging.logging(traces)
        except:
            print("\nProblem with logging\n")

        self.params[f"call_{self.call_num}"] = {
            'datasets': self.datasets,
            'h_params': {
                'criterion': f'{self.criterion}',
                'lr': f'{lr}',
                'wd': f'{wd}',
                'l1': f'{l1}',
                'bs': f'{bs}',
            },
            'scores': {
                'best_epoch': f'{best_epoch}',
                'train_loss': f'{train_loss[int(best_epoch)]}',
                'valid_loss': f'{valid_loss}',
                'test_loss': f'{test_loss}',
                'train_acc': f'{train_acc[int(best_epoch)]}',
                'valid_acc': f'{valid_acc}',
                'test_acc': f'{test_acc}',
                'train_mcc': f'{train_mcc[int(best_epoch)]}',
                'valid_mcc': f'{valid_mcc}',
                'test_mcc': f'{test_mcc}',
            }
        }
        json.dump(self.params, open(f'{file_path}/params.json', 'w'))

        return 1 - np.mean(traces['valid']['accuracies'])

    def test(self, params):
        pass


if __name__ == "__main__":
    import argparse

    seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--intensities_csv", type=str, default="data/output_200.csv",
                        help="Path to intensities csv file")
    parser.add_argument("--verbose", type=int, default=2,
                        help="Path to labels csv file")
    parser.add_argument("--freeze", type=int, default=1,
                        help="Path to labels csv file")
    parser.add_argument("--model", type=str, default='lenet',
                        help="Name of the model to use [lenet, lecun, vgg9 or logistic].")
    parser.add_argument("--criterion", type=str, default='categorical_crossentropy',
                        help=".")
    parser.add_argument("--pretrained_path", type=str, default=None,  #
                        help="")
    parser.add_argument("--cumulative_step", type=int, default=0,  #
                        help="")
    parser.add_argument("--scaler", type=str, default='normalizer',  #
                        help="")

    args = parser.parse_args()

    if args.model in ['lenet', 'lecun', 'vgg9']:
        model = CNN
    elif args.model in ['logistic']:
        model = Linear
    else:
        exit("This model is not implemented, chose lenet, lecun, vgg9 or logistic ")
    if args.freeze == 0:
        args.freeze = False
    elif args.freeze == 1:
        args.freeze = True
    else:
        exit('freeze should be 0 or 1')

    train = Train(args.intensities_csv,
                  criterion=args.criterion,
                  variant=args.model,
                  verbose=args.verbose,
                  model_name=model,
                  model_path=args.pretrained_path,
                  freeze=args.freeze,
                  scaler=args.scaler
                  )

    # TODO epochs not needed with validation_data in fit()
    space = [
        # Integer(1, 100, "uniform", name='epochs'),
        Real(1e-6, 1e-3, "log-uniform", name='lr'),
        Real(1e-12, 1e-1, "log-uniform", name='wd'),
        Real(1e-12, 1e-1, "log-uniform", name='l1'),
        Integer(1, 4, "uniform", name='bs'),
    ]

    test_mean = gp_minimize(train.train, space, n_calls=100, random_state=42)
    print(test_mean)

    # TODO Ajouter une etape pour effacer tous les moins bon modeles
