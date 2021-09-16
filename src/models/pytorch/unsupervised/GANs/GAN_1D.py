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


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self,
                 in_channel,
                 channel,
                 n_res_block,
                 n_res_channel):
        super().__init__()

        blocks = [
            nn.Conv1d(in_channel, channel // 2, 16, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // 2, channel, 8, stride=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel, channel, 6, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel, channel, 6, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel, channel, 6, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel, channel, 6, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel, channel, 6, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel, channel, 6, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel, channel, 6, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel, channel, 6, stride=2, padding=0),
        ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Generator(Base):
    def __init__(self,
                 dropout=0.5,
                 input_size=13181,
                 z_dim=100
                 ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, input_size),
        )

        self.random_init()

    def random_init(self, init_func=torch.nn.init.xavier_uniform_):
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        # x[x > 1] = 1.0
        return x

    def get_model_name(self):
        return 'generator'


class Discriminator(Base):
    def __init__(
            self,
            dropout=0,
            input_size=13181
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1),
        )

        self.random_init()

    def random_init(self, init_func=torch.nn.init.xavier_uniform_):
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.linear(x.squeeze())
        x = torch.sigmoid(x)
        return x

    def get_model_name(self):
        return 'discriminator'
