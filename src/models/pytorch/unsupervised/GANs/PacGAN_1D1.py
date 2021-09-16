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
                 norm,
                 n_res_block,
                 n_res_channel
                 ):
        super().__init__()
        # nn.BatchNorm1d = norm
        self.in_channel = in_channel
        self.channel = channel
        if norm == 'bn':
            self.get_blocks_bn_mp()
        elif norm == 'bn3':
            self.get_blocks_bn_mp()
        elif norm == 'sn':
            self.get_blocks_sn_mp()
        elif norm == 'ln':
            self.get_blocks_ln()
        else:
            print('WARNING: NO NORMALIZATION')
            exit()

        # for i in range(n_res_block):
        #     blocks.append(ResBlock(channel, n_res_channel))

        # blocks.append(nn.ReLU(inplace=True))

        # self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        hidden_values = []
        for i, block in enumerate(self.blocks):
            if i > 0:
                hidden_values += [x]
            x = block(x)

        return x, hidden_values

    def get_blocks_bn_msg(self):
        self.blocks = nn.ModuleList()
        self.blocks += [nn.Sequential(
            nn.Conv1d(self.in_channel, self.channel, 42, stride=1, padding=0, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True)
        )]
        self.blocks += [nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 256, stride=1, padding=0, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True)
        )]
        self.blocks += [nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 512, stride=1, padding=0, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
        )]
        self.blocks += [nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 128, stride=2, padding=0, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
        )]
        self.blocks += [nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 128, stride=2, padding=0, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
        )]
        self.blocks += [nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 33, stride=1, padding=0, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
        )]

    def get_blocks_sn_msg(self):
        self.blocks = nn.ModuleList()
        self.blocks += [nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv1d(self.in_channel, self.channel, 42, stride=2, padding=42, dilation=1)
            ),
            nn.LeakyReLU(inplace=True)
        )]
        self.blocks += [nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv1d(self.channel, self.channel, 256, stride=2, padding=256, dilation=1),
            ),
            nn.LeakyReLU(inplace=True)
        )]
        self.blocks += [nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv1d(self.channel, self.channel, 512, stride=2, padding=512, dilation=1),
            ),
            nn.LeakyReLU(inplace=True),
        )]
        self.blocks += [nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv1d(self.channel, self.channel, 512, stride=1, padding=16, dilation=1),
            ),
            nn.LeakyReLU(inplace=True),
        )]
        self.blocks += [nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv1d(self.channel, self.channel, 12, stride=1, padding=0, dilation=1),
            ),
            nn.LeakyReLU(inplace=True),
        )]
        # self.blocks += [nn.Sequential(
        #     nn.utils.spectral_norm(
        #         nn.Conv1d(self.channel, self.channel, 128, stride=2, padding=0, dilation=1),
        #     ),
        #     nn.LeakyReLU(inplace=True),
        # )]
        # self.blocks += [nn.Sequential(
        #     nn.utils.spectral_norm(
        #         nn.Conv1d(self.channel, self.channel, 33, stride=1, padding=0, dilation=1),
        #     ),
        #     nn.LeakyReLU(inplace=True),
        # )]

    def get_blocks_sn(self):
        self.blocks = nn.ModuleList()
        self.blocks += nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv1d(self.in_channel, self.channel // 2, 16, stride=4, padding=8)
            ),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks += nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv1d(self.channel // 2, self.channel, 8, stride=3, padding=1)
            ),
            nn.LeakyReLU(inplace=True),
        )

        self.blocks += nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1)
            ),
            nn.LeakyReLU(inplace=True),
        )

        self.blocks += nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1)
            ),
            nn.LeakyReLU(inplace=True),
        )

        self.blocks += nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1)
            ),
            nn.LeakyReLU(inplace=True),
        )

        self.blocks += nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1)
            ),
            nn.LeakyReLU(inplace=True),
        )

        self.blocks += nn.Sequential(
            # nn.utils.spectral_norm(
            # nn.utils.spectral_norm(
            nn.Conv1d(self.channel, self.channel, 4, stride=1, padding=0)
            # ),
            # nn.LeakyReLU(inplace=True),
            # ),
        )

    def get_blocks_bn(self):
        self.blocks = nn.ModuleList()
        self.blocks += nn.Sequential(
            nn.Conv1d(self.in_channel, self.channel // 2, 16, stride=4, padding=8),
            nn.BatchNorm1d(num_features=self.channel // 2),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel // 2, self.channel, 8, stride=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
        )

        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
        )

        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
        )

        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
        )

        self.blocks += nn.Sequential(
            # nn.utils.spectral_norm(
            # nn.utils.spectral_norm(
            nn.Conv1d(self.channel, self.channel, 4, stride=1, padding=0)
            # ),
            # nn.LeakyReLU(inplace=True),
            # ),
        )

    def get_blocks_bn_mp(self):
        self.blocks = nn.ModuleList()
        self.blocks += nn.Sequential(
            nn.Conv1d(self.in_channel, self.channel // 2, 8, stride=2, padding=8),
            nn.BatchNorm1d(num_features=self.channel // 2),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel // 2, self.channel, 8, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 6, stride=1, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 3, stride=1, padding=0),
        )

    def get_blocks_bn_mp_avg(self):
        self.blocks = nn.ModuleList()
        self.blocks += nn.Sequential(
            nn.Conv1d(self.in_channel, self.channel // 2, 8, stride=2, padding=8),
            nn.BatchNorm1d(num_features=self.channel // 2),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel // 2, self.channel, 8, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool1d(2),
        )

        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool1d(2),
        )

        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool1d(2),
        )
        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 6, stride=1, padding=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool1d(2),
        )

        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 3, stride=1, padding=0),
            # nn.BatchNorm1d(num_features=self.channel),
            # nn.LeakyReLU(inplace=True),
        )

    def get_blocks_sn_mp(self):
        self.blocks = nn.ModuleList()
        self.blocks += nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(self.in_channel, self.channel // 2, 8, stride=2, padding=8)),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks += nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(self.channel // 2, self.channel, 8, stride=2, padding=1)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        self.blocks += nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        self.blocks += nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.blocks += nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(self.channel, self.channel, 6, stride=1, padding=1)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 3, stride=1, padding=0),
        )

    def get_blocks_ln(self):
        self.blocks = nn.ModuleList()
        self.blocks += nn.Sequential(
            nn.Conv1d(self.in_channel, self.channel // 2, 16, stride=4, padding=8),
            nn.LayerNorm([self.channel // 2, 330]),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel // 2, self.channel, 8, stride=3, padding=1),
            nn.LayerNorm([self.channel, 109]),
            nn.LeakyReLU(inplace=True),
        )

        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.LayerNorm([self.channel, 53]),
            nn.LeakyReLU(inplace=True),
        )

        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.LayerNorm([self.channel, 25]),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.LayerNorm([self.channel, 11]),
            nn.LeakyReLU(inplace=True),
        )

        self.blocks += nn.Sequential(
            nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1),
            nn.LayerNorm([self.channel, 4]),
            nn.LeakyReLU(inplace=True),
        )

        self.blocks += nn.Sequential(
            # nn.utils.spectral_norm(
            # nn.utils.spectral_norm(
            nn.Conv1d(self.channel, self.channel, 4, stride=1, padding=0)
            # ),
            # nn.LeakyReLU(inplace=True),
            # ),
        )

    def get_blocks_bn_sm(self):
        blocks = [
            nn.Conv1d(self.in_channel, self.channel // 2, 16, stride=4, padding=8),
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

            nn.Conv1d(self.channel, self.channel, 4, stride=1, padding=0),
            # nn.BatchNorm1d(num_features=self.channel),
            # nn.LeakyReLU(inplace=True),
        ]

        return blocks

    def get_blocks_sn_sm(self):
        blocks = [
            nn.utils.spectral_norm(
                nn.Conv1d(self.in_channel, self.channel // 2, 16, stride=4, padding=8)
            ),
            nn.LeakyReLU(inplace=True),

            nn.utils.spectral_norm(
                nn.Conv1d(self.channel // 2, self.channel, 8, stride=3, padding=1)
            ),
            nn.LeakyReLU(inplace=True),

            nn.utils.spectral_norm(
                nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1)
            ),
            nn.LeakyReLU(inplace=True),

            nn.utils.spectral_norm(
                nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1)
            ),
            nn.LeakyReLU(inplace=True),

            nn.utils.spectral_norm(
                nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1)
            ),
            nn.LeakyReLU(inplace=True),

            nn.utils.spectral_norm(
                nn.Conv1d(self.channel, self.channel, 6, stride=2, padding=1)
            ),
            nn.LeakyReLU(inplace=True),

            # nn.utils.spectral_norm(
            # nn.utils.spectral_norm(
            nn.Conv1d(self.channel, self.channel, 4, stride=1, padding=0)
            # ),
            # nn.LeakyReLU(inplace=True),
            # ),
        ]

        return blocks


class Decoder(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 channel,
                 norm,
                 n_res_block,
                 n_res_channel,
                 size='large_cnn'
                 ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.channel = channel

        # blocks = [nn.Conv1d(in_channel, channel, 3, padding=1)]

        # for i in range(n_res_block):
        #     blocks.append(ResBlock(channel, n_res_channel))

        # blocks.append(nn.LeakyReLU(inplace=True))

        if size == 'large_cnn':
            if norm == 'bn' or norm == 'bn2':
                self.build_large_cnn_bn_mup2()
            elif norm == 'bn3':
                self.build_large_cnn_bn_mup3()
            elif norm == 'sn':
                self.build_large_cnn_bn_mup2()
            elif norm == 'ln':
                self.build_large_cnn_ln()

        elif size == 'small_cnn':
            self.build_small_cnn()
        else:
            self.build_smaller_cnn()
        # self.blocks = nn.Sequential(*blocks)

    def build_large_cnn(self):
        blocks = [
            nn.ConvTranspose1d(self.in_channel, self.channel, 512, stride=1, padding=0, dilation=1),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 512, stride=1, padding=0, dilation=1),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 256, stride=1, padding=0, dilation=1),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.out_channel, 42, stride=1, padding=0, dilation=1),
            # nn.BatchNorm1d(num_features=self.out_channel),
            # nn.LeakyReLU(inplace=True),
        ]
        return blocks

    def build_large_cnn_bn(self):
        self.blocks = nn.ModuleList()
        self.blocks += nn.Sequential(
            nn.ConvTranspose1d(self.in_channel, self.channel, 512, stride=1, padding=0, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True)
        )
        self.blocks += nn.Sequential(
            nn.ConvTranspose1d(self.channel, self.channel, 512, stride=1, padding=0, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks += nn.Sequential(
            nn.ConvTranspose1d(self.channel, self.channel, 256, stride=1, padding=0, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks += nn.Sequential(
            nn.ConvTranspose1d(self.channel, self.out_channel, 42, stride=1, padding=0, dilation=1),
        )

    def build_large_cnn_bn_mup2(self):
        self.blocks = nn.ModuleList()
        self.blocks += nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(self.in_channel, self.channel, 6, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True)
        )
        self.blocks += nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(self.channel, self.channel, 6, stride=2, padding=3, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True)
        )
        self.blocks += nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(self.channel, self.channel, 6, stride=2, padding=4, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks += nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(self.channel, self.channel, 8, stride=2, padding=6, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks += nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(self.channel, self.out_channel, 8, stride=2, padding=4, dilation=1),
        )

    def build_large_cnn_bn_mup3(self):
        self.blocks = nn.ModuleList()
        self.blocks += nn.Sequential(
            # nn.Upsample(scale_factor=3),
            nn.ConvTranspose1d(self.in_channel, self.channel, 8, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True)
        )
        self.blocks += nn.Sequential(
            nn.Upsample(scale_factor=3),
            nn.ConvTranspose1d(self.channel, self.channel, 7, stride=2, padding=2, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True)
        )
        self.blocks += nn.Sequential(
            nn.Upsample(scale_factor=3),
            nn.ConvTranspose1d(self.channel, self.channel, 8, stride=2, padding=4, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks += nn.Sequential(
            nn.Upsample(scale_factor=3),
            nn.ConvTranspose1d(self.channel, self.out_channel, 8, stride=2, padding=4, dilation=1),
        )

    def build_large_cnn_ln(self):
        self.blocks = nn.ModuleList()
        self.blocks += [nn.Sequential(
            nn.ConvTranspose1d(self.in_channel, self.channel, 512, stride=1, padding=0, dilation=1),
            nn.LayerNorm(512),
            nn.LeakyReLU(inplace=True)
        )]
        self.blocks += [nn.Sequential(
            nn.ConvTranspose1d(self.channel, self.channel, 512, stride=1, padding=0, dilation=1),
            nn.LayerNorm(1023),
            nn.LeakyReLU(inplace=True),
        )]
        self.blocks += [nn.Sequential(
            nn.ConvTranspose1d(self.channel, self.channel, 256, stride=1, padding=0, dilation=1),
            nn.LayerNorm(1278),
            nn.LeakyReLU(inplace=True),
        )]
        self.blocks += [nn.Sequential(
            nn.ConvTranspose1d(self.channel, self.out_channel, 42, stride=1, padding=0, dilation=1),
        )]

    def build_large_cnn_sn(self):
        self.blocks = nn.Sequential()
        self.blocks += nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.in_channel, self.channel, 512, stride=1, padding=0, dilation=1)
            ),
            nn.LeakyReLU(inplace=True)
        )
        self.blocks += nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.channel, self.channel, 512, stride=1, padding=0, dilation=1)
            ),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks += nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.channel, self.channel, 256, stride=1, padding=0, dilation=1)
            ),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks += nn.Sequential(
            nn.ConvTranspose1d(self.channel, self.out_channel, 42, stride=1, padding=0, dilation=1),
        )

    def build_small_cnn(self):
        blocks = [
            nn.ConvTranspose1d(self.in_channel, self.channel, 64, stride=4, padding=2, dilation=1),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 32, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 16, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 16, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 16, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 16, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 16, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 16, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.out_channel, 12, stride=1, padding=0, dilation=1),
        ]
        return blocks

    def build_small_cnn_bn(self):
        blocks = [
            nn.ConvTranspose1d(self.in_channel, self.channel, 64, stride=4, padding=2, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 32, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 16, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 16, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 16, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 16, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 16, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 16, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.out_channel, 12, stride=1, padding=0, dilation=1),
            # nn.BatchNorm1d(num_features=self.out_channel),
            # nn.LeakyReLU(inplace=True),
        ]
        return blocks

    def build_small_cnn_sn(self):
        blocks = [
            nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.in_channel, self.channel, 64, stride=4, padding=2, dilation=1)
            ),
            nn.LeakyReLU(inplace=True),

            nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.channel, self.channel, 32, stride=2, padding=1, dilation=1)
            ),
            nn.LeakyReLU(inplace=True),

            nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.channel, self.channel, 16, stride=2, padding=1, dilation=1)
            ),
            nn.LeakyReLU(inplace=True),

            nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.channel, self.channel, 16, stride=2, padding=1, dilation=1)
            ),
            nn.LeakyReLU(inplace=True),

            nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.channel, self.channel, 16, stride=2, padding=1, dilation=1)
            ),
            nn.LeakyReLU(inplace=True),

            nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.channel, self.channel, 16, stride=1, padding=1, dilation=1)
            ),
            nn.LeakyReLU(inplace=True),

            nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.channel, self.channel, 16, stride=1, padding=1, dilation=1)
            ),
            nn.LeakyReLU(inplace=True),

            nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.channel, self.channel, 16, stride=1, padding=1, dilation=1)
            ),
            nn.LeakyReLU(inplace=True),

            nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.channel, self.out_channel, 12, stride=1, padding=0, dilation=1)
            )
        ]
        self.blocks = nn.Sequential(*blocks)

    def build_smaller_cnn(self):
        blocks = [
            nn.ConvTranspose1d(self.in_channel, self.channel, 16, stride=4, padding=2, dilation=4),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 16, stride=2, padding=1, dilation=2),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 16, stride=2, padding=1, dilation=2),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 16, stride=2, padding=1, dilation=2),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 12, stride=2, padding=1, dilation=2),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.out_channel, 10, stride=1, padding=0, dilation=1),
            # nn.BatchNorm1d(num_features=self.out_channel),
            # nn.LeakyReLU(inplace=True),
        ]
        return blocks

    def build_smaller_cnn_bn(self):
        blocks = [
            nn.ConvTranspose1d(self.in_channel, self.channel, 16, stride=4, padding=2, dilation=4),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 16, stride=2, padding=1, dilation=2),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 16, stride=2, padding=1, dilation=2),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 16, stride=2, padding=1, dilation=2),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.channel, 12, stride=2, padding=1, dilation=2),
            nn.BatchNorm1d(num_features=self.channel),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(self.channel, self.out_channel, 10, stride=1, padding=0, dilation=1),
            # nn.BatchNorm1d(num_features=self.out_channel),
            # nn.LeakyReLU(inplace=True),
        ]
        return blocks

    def build_smaller_cnn_sn(self):
        blocks = [
            nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.in_channel, self.channel, 16, stride=4, padding=2, dilation=4)
            ),
            nn.LeakyReLU(inplace=True),

            nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.channel, self.channel, 16, stride=2, padding=1, dilation=2)
            ),
            nn.LeakyReLU(inplace=True),


            nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.channel, self.channel, 16, stride=2, padding=1, dilation=2)
            ),
            nn.LeakyReLU(inplace=True),


            nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.channel, self.channel, 16, stride=2, padding=1, dilation=2)
            ),
            nn.LeakyReLU(inplace=True),

            nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.channel, self.channel, 12, stride=2, padding=1, dilation=2)
            ),
            nn.LeakyReLU(inplace=True),

            nn.utils.spectral_norm(
                nn.ConvTranspose1d(self.channel, self.out_channel, 10, stride=1, padding=0, dilation=1)
            )
        ]
        return blocks

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Generator(Base):
    def __init__(self,
                 in_channel=1,
                 channel=128,
                 embed_dim=64,
                 dropout=0.5,
                 out_activation=nn.Sigmoid,
                 norm='bn'
                 ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dec = Decoder(
            embed_dim,
            in_channel,
            channel,
            norm,
            n_res_block=5,
            n_res_channel=32,
        )
        if out_activation is not None:
            self.activation = out_activation()
        else:
            self.activation = None
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
            channel=128,
            n_res_block=5,
            n_res_channel=32,
            dropout=0.5,
            norm='bn'
    ):
        super().__init__()
        if out_activation is not None:
            self.activation = out_activation()
        else:
            self.activation = None
        self.dropout = nn.Dropout(p=dropout)
        self.enc = Encoder(
            in_channel,
            channel,
            norm,
            n_res_block,
            n_res_channel,
        )

        self.linear1 = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(num_features=channel),
            nn.ReLU(),
        )
        self.linear2 = nn.Linear(channel, 1)
        self.random_init()

    def random_init(self, init_func=torch.nn.init.kaiming_normal_):
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x, hidden_values = self.enc(x)
        x = x.squeeze()
        # x = self.linear1(x)
        x = self.linear2(x)
        if self.activation is not None:
            x = self.activation(x)
        return x, hidden_values

    def meta_update(self, dis, meta_lr=0.01):
        for m, q in zip(self.state_dict(), dis.state_dict()):
            if 'bias' in m or 'weight' in m:
                self.state_dict()[m] += (self.state_dict()[m] - dis.state_dict()[m]) * meta_lr

    def get_model_name(self):
        return 'discriminator'
