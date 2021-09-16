#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Dropout, BatchNormalization, MaxPooling1D, LeakyReLU
from tensorflow.keras.regularizers import l1_l2
from src.models.keras.Base import Base


class CNN(Base):
    def __init__(self, h_params, nb_classes, batch_size, variant='lecun', activation='relu'):
        super(CNN, self).__init__(h_params, nb_classes, batch_size, variant, activation)

    def build(self, input_shape):
        self.input_shape = input_shape
        if self.variant == 'lecun':
            self.lecun()
        elif self.variant == 'lenet':
            self.lenet()
        elif self.variant == 'vgg9':
            self.vgg9()
        else:
            exit(f'Model {self.variant} unrecognized.\n Accepted variants: lecun, lenet and vgg9')

    def get_model_name(self):
        return "CNN"

    def lecun(self):
        try:
            assert self.nb_classes is not None
        except ValueError:
            exit("Must set the number of classes first")
        self.model = Sequential([
            Conv1D(filters=6, kernel_size=21, strides=1, padding='same', activation='relu',
                   input_shape=self.input_shape, kernel_initializer=keras.initializers.he_normal(), activity_regularizer=l1_l2(self.l1, self.wd)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Conv1D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', activity_regularizer=l1_l2(self.l1, self.wd)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Flatten(),
            Dense(120, activation='relu', activity_regularizer=l1_l2(self.l1, self.wd)),
            Dense(84, activity_regularizer=l1_l2(self.l1, self.wd)),
            Dense(self.nb_classes, activation='softmax', activity_regularizer=l1_l2(self.l1, self.wd))  # or Activation('softmax')
        ])

    def lenet(self):
        self.model = Sequential([
            Conv1D(filters=16, kernel_size=21, strides=1, padding='same', input_shape=self.input_shape,
                   kernel_initializer=keras.initializers.he_normal(), activity_regularizer=l1_l2(self.l1, self.wd)),
            BatchNormalization(),
            LeakyReLU(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Conv1D(filters=32, kernel_size=11, strides=1, padding='same', activity_regularizer=l1_l2(self.l1, self.wd)),
            BatchNormalization(),
            LeakyReLU(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activity_regularizer=l1_l2(self.l1, self.wd)),
            BatchNormalization(),
            LeakyReLU(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Flatten(),
            Dense(2050, activation='relu', activity_regularizer=l1_l2(self.l1, self.wd)),
            Dropout(0.5),
            Dense(self.nb_classes, activation='softmax', activity_regularizer=l1_l2(self.l1, self.wd))  # or Activation('softmax')
        ])

    def vgg9(self):
        self.model = Sequential([
            Conv1D(filters=64, kernel_size=21, strides=1, padding='same', activation='relu',
                   input_shape=self.input_shape, kernel_initializer=keras.initializers.he_normal(), activity_regularizer=l1_l2(self.l1, self.wd)),
            BatchNormalization(),
            Conv1D(filters=64, kernel_size=21, strides=1, padding='same', activation='relu', activity_regularizer=l1_l2(self.l1, self.wd)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Conv1D(filters=128, kernel_size=11, strides=1, padding='same', activation='relu', activity_regularizer=l1_l2(self.l1, self.wd)),
            BatchNormalization(),
            Conv1D(filters=128, kernel_size=11, strides=1, padding='same', activation='relu', activity_regularizer=l1_l2(self.l1, self.wd)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Conv1D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu', activity_regularizer=l1_l2(self.l1, self.wd)),
            BatchNormalization(),
            Conv1D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu', activity_regularizer=l1_l2(self.l1, self.wd)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Flatten(),
            Dense(4096, activation='relu', activity_regularizer=l1_l2(self.l1, self.wd)),
            Dropout(0.5),
            Dense(4096, activation='relu', activity_regularizer=l1_l2(self.l1, self.wd)),
            Dropout(0.5),
            Dense(self.nb_classes, activation='softmax', activity_regularizer=l1_l2(self.l1, self.wd))  # or Activation('softmax')
        ])
