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
                 n_res_channel,
                 bin_size="1"):
        super().__init__()

        self.in_channel = in_channel
        self.channel = channel
        self.bin_size = bin_size
        if self.bin_size == "01":
            blocks = self.build_01()
        elif self.bin_size == "1":
            blocks = self.build_1()
        else:
            exit(f"No build available for bin size: {self.bin_size}")


        # for i in range(n_res_block):
        #     blocks.append(ResBlock(channel, n_res_channel))

        # blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def build_01(self):
        blocks = [
            nn.Conv1d(self.in_channel, self.channel // 2, 16, stride=4, padding=1),
            nn.BatchNorm1d(num_features=self.channel // 2),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(self.channel // 2, self.channel, 8, stride=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.BatchNorm1d(num_features=self.channel),
            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=0),
        ]
        return blocks

    def build_1(self):
        blocks = [
            nn.Conv1d(self.in_channel, self.channel // 2, 16, stride=4, padding=1),
            nn.BatchNorm1d(num_features=self.channel // 2),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(self.channel // 2, self.channel, 8, stride=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.BatchNorm1d(num_features=self.channel),
            nn.Conv1d(self.channel, self.channel, 4, stride=1, padding=0),
        ]
        return blocks

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 channel,
                 n_res_block,
                 n_res_channel,
                 bin_size="1"
                 ):
        super().__init__()

        # blocks = [nn.Conv1d(in_channel, channel, 3, padding=1)]

        # for i in range(n_res_block):
        #     blocks.append(ResBlock(channel, n_res_channel))

        # blocks.append(nn.LeakyReLU(inplace=True))
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.channel = channel
        self.bin_size = bin_size
        if self.bin_size == "01":
            blocks = self.build_01()
        elif self.bin_size == "1":
            blocks = self.build_1()
        else:
            exit(f"No build available for bin size: {self.bin_size}")

        self.blocks = nn.Sequential(*blocks)

    def build_01(self):
        blocks = [
            nn.ConvTranspose1d(self.in_channel, self.channel, 1024, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 512, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 256, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 128, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 64, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 32, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            # nn.BatchNorm1d(num_features=self.channel),
            nn.ConvTranspose1d(self.channel, self.out_channel, 6, stride=1, padding=1, dilation=1),
        ]
        return blocks

    def build_1(self):
        blocks = [
            nn.ConvTranspose1d(self.in_channel, self.channel, 512, stride=1, padding=0, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 512, stride=1, padding=0, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 256, stride=1, padding=0, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.out_channel, 42, stride=1, padding=0, dilation=1),
            # nn.BatchNorm1d(num_features=self.channel),
        ]
        return blocks

    def forward(self, input):
        return self.blocks(input)


class Generator(Base):
    def __init__(self,
                 out_activation,
                 in_channel=1,
                 channel=128,
                 embed_dim=64,
                 dropout=0,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dec = Decoder(
            embed_dim,
            in_channel,
            channel,
            n_res_block=5,
            n_res_channel=32,
        )
        if out_activation is not None:
            self.out_activation = out_activation()
        else:
            self.out_activation = None
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
        x = self.dec(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

    def get_model_name(self):
        return 'generator'


class Discriminator(Base):
    def __init__(
            self,
            out_activation,
            in_channel=1,
            channel=128,
            n_res_block=5,
            n_res_channel=32,
            dropout=0,
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.enc = Encoder(
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
        )
        if out_activation is not None:
            self.out_activation = out_activation()
        else:
            self.out_activation = None

        self.linear = nn.Linear(channel, 1)
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
        x = self.linear(x)

        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

    def get_model_name(self):
        return 'discriminator'
