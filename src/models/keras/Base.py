#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

from tensorflow.keras.layers import Dense, Layer, Flatten, Conv1D, Dropout, BatchNormalization, MaxPooling1D, LeakyReLU


class Base(Layer):
    def __init__(self, h_params, nb_classes, batch_size, variant, activation):
        super(Base, self).__init__()

        self.model = None
        self.__previous_models = None
        self.__input_shape = None

        self.__h_params = h_params
        self.__activation = activation
        self.__variant = variant
        self.__nb_classes = nb_classes
        self.__batch_size = batch_size

    def build(self, input_shape):
        exit('The function `build` needs to be implemented. This is an abstract class.')

    def get_model_name(self):
        exit('`get_model_name` needs to be implemented. This is an abstract class.')

    @property
    def variant(self):
        return self.__variant

    @property
    def nb_classes(self):
        return self.__nb_classes

    @property
    def input_shape(self):
        return self.__input_shape

    @property
    def activation(self):
        return self.__activation

    @property
    def h_params(self):
        return self.__h_params

    @property
    def n_epochs(self):
        return self.__h_params['n_epochs']

    @property
    def bs(self):
        return self.__h_params['bs']

    @property
    def wd(self):
        return self.__h_params['wd']

    @property
    def l1(self):
        return self.__h_params['l1']


    @variant.setter
    def variant(self, variant):
        self.__variant = variant

    @bs.setter
    def bs(self, bs):
        self.__h_params['bs'] = bs

    @nb_classes.setter
    def nb_classes(self, nb_classes):
        self.__nb_classes = nb_classes

    @input_shape.setter
    def input_shape(self, input_shape):
        self.__input_shape = input_shape

    @activation.setter
    def activation(self, activation):
        self.__activation = activation

    @h_params.setter
    def h_params(self, h_params):
        self.__h_params = h_params

    @n_epochs.setter
    def n_epochs(self, n_epochs):
        self.__h_params['n_epochs'] = n_epochs

    @wd.setter
    def wd(self, wd):
        self.__h_params['wd'] = wd

    @l1.setter
    def l1(self, l1):
        self.__h_params['l1'] = l1

