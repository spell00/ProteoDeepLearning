#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.regularizers import l1_l2
from src.models.keras.Base import Base


class Linear(Base):
    def __init__(self, nb_classes, input_shape, wd, l1=0, variant='logistic', activation='relu'):
        super(Linear, self).__init__(nb_classes, input_shape, wd, variant, activation, l1)
        self.build(self.variant)

    def build(self, variant):
        if variant == 'logistic':
            self.logistic()
        else:
            exit(f'Model {variant} unrecognized.\n Only variant: logistic')

    def logistic(self):
        self.model = Sequential([
            Flatten(),
            Dense(self.nb_classes, activation='softmax', activity_regularizer=l1_l2(self.l1, self.wd))  # or Activation('softmax')
        ])

    def get_model_name(self):
        return "Linear"
