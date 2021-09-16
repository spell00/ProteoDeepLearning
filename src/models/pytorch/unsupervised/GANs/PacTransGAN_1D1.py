import torch
from torch import nn
from src.models.Base import Base


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import dataset


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModelEncoder(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # src = self.encoder(src) * math.sqrt(self.d_model)
        # src = self.pos_encoder(src.unsqueeze(1))
        output = self.transformer_encoder(src.unsqueeze(1))
        return output


class TransformerModelDecoder(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerDecoder(encoder_layers, nlayers)
        self.d_model = d_model


    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class Generator(Base):
    def __init__(self,
                 in_channel=1,
                 d_model=64,
                 d_hid=128,
                 nhead=8,
                 dropout=0,
                 out_activation=nn.Sigmoid,
                 norm='bn'
                 ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dec = TransformerModelEncoder(d_model, nhead, d_hid, in_channel, dropout)
        if out_activation is not None:
            self.activation = out_activation()
        else:
            self.activation = None
        self.src_mask = generate_square_subsequent_mask(d_model)
        self.random_init()

    def random_init(self, init_func=torch.nn.init.kaiming_normal_):
        torch.manual_seed(42)
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        # src_mask unused
        x = self.dec(x, self.src_mask)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def meta_update(self, gen, meta_lr=0.01):
        for m, q in zip(self.modules(), gen.modules()):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                m.weight.data += (m.weight.data - q.weight.data) * meta_lr
                if m.bias is not None:
                    m.bias.data += (m.bias.data - q.bias.data) * meta_lr

    def get_model_name(self):
        return 'generator'


class Discriminator(Base):
    def __init__(
            self,
            out_activation,
            in_channel=1,
            d_model=64,
            d_hid=128,
            nhead=8,
            dropout=0,
            norm='bn'
    ):
        super().__init__()
        if out_activation is not None:
            self.activation = out_activation()
        else:
            self.activation = None
        self.dropout = nn.Dropout(p=dropout)
        self.enc = TransformerModelDecoder(d_model, nhead, d_hid, in_channel, dropout)

        self.linear1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(num_features=d_model),
            nn.ReLU(),
        )
        self.linear2 = nn.Linear(d_model, 1)
        self.random_init()

    def random_init(self, init_func=torch.nn.init.kaiming_normal_):
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.enc(x).squeeze()
        # x = self.linear1(x)
        x = self.linear2(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def meta_update(self, dis, meta_lr=0.01):
        for m, q in zip(self.state_dict(), dis.state_dict()):
            if 'bias' in m or 'weight' in m:
                self.state_dict()[m] += (self.state_dict()[m] - dis.state_dict()[m]) * meta_lr

    def get_model_name(self):
        return 'discriminator'
