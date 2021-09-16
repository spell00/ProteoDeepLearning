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
                 channel
                 ):
        super().__init__()

        blocks = [
            nn.Conv1d(in_channel, channel // 2, 16, stride=4, padding=1),
            nn.BatchNorm1d(num_features=channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(channel // 2, channel, 8, stride=3, padding=1),
            nn.BatchNorm1d(num_features=channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(channel, channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(channel, channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(channel, channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(channel, channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(channel, channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(channel, channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(channel, channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=channel),
            nn.LeakyReLU(inplace=True),

            nn.BatchNorm1d(num_features=channel),
            nn.Conv1d(channel, channel, 6, stride=2, padding=0),
        ]

        # for i in range(n_res_block):
        #     blocks.append(ResBlock(channel, n_res_channel))

        # blocks.append(nn.LeakyReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
            self, in_channel, out_channel, channel
    ):
        super().__init__()

        blocks = [
            nn.ConvTranspose1d(in_channel, channel, 6, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(num_features=channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(channel, channel, 6, stride=1, padding=1, dilation=2),
            nn.BatchNorm1d(num_features=channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(channel, channel, 6, stride=1, padding=1, dilation=4),
            nn.BatchNorm1d(num_features=channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(channel, channel, 6, stride=1, padding=1, dilation=8),
            nn.BatchNorm1d(num_features=channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(channel, channel, 6, stride=1, padding=1, dilation=16),
            nn.BatchNorm1d(num_features=channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(channel, channel, 6, stride=1, padding=1, dilation=32),
            nn.BatchNorm1d(num_features=channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(channel, channel, 6, stride=2, padding=1, dilation=64),
            nn.BatchNorm1d(num_features=channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(channel, channel, 5, stride=2, padding=1, dilation=128),
            nn.BatchNorm1d(num_features=channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(channel, channel, 5, stride=2, padding=40, dilation=256),
            nn.BatchNorm1d(num_features=channel),
            nn.LeakyReLU(inplace=True),

            nn.BatchNorm1d(num_features=channel),
            nn.ConvTranspose1d(channel, out_channel, 5, stride=2, padding=94, dilation=512),
        ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Generator(Base):
    def __init__(self,
                 in_channel=1,
                 channel=128,
                 n_res_block=2,
                 n_res_channel=32,
                 embed_dim=64,
                 dropout=0,
                 decay=0.99
                 ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.dec = Decoder(
            embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        self.random_init()

    def random_init(self, init_func=torch.nn.init.kaiming_uniform_):
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.dec(x)
        # x = torch.tanh(x)
        x = torch.sigmoid(x)
        return x

    def get_model_name(self):
        return 'generator'


class Discriminator(Base):
    def __init__(
            self,
            in_channel=1,
            channel=128,
            n_res_block=2,
            n_res_channel=32,
            embed_dim=64,
            dropout=0,
            decay=0.99
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.enc = Encoder(
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
        )

        self.linear = nn.Linear(channel, 1)
        self.random_init()

    def random_init(self, init_func=torch.nn.init.kaiming_uniform_):
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.enc(x).squeeze()
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

    def get_model_name(self):
        return 'discriminator'
