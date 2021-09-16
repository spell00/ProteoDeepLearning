import torch
from torch import nn
from torch.nn import functional as F
from src.models.Base import Base
import src.models.pytorch.utils.distributed as dist_fn


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


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        torch.manual_seed(42)
        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        # embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


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
    def __init__(self, in_channel, channel, n_res_block, n_res_channel):
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


class Decoder(nn.Module):
    def __init__(
            self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv1d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(n_res_channel, channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose1d(channel, channel, 6, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose1d(channel, channel, 6, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose1d(channel, channel, 6, stride=2, padding=2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose1d(channel, channel, 6, stride=2, padding=2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose1d(channel, channel, 6, stride=2, padding=3),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose1d(channel, channel, 6, stride=2, padding=4),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose1d(channel, channel, 6, stride=2, padding=4),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose1d(channel, channel, 6, stride=2, padding=6),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose1d(
                        channel, channel // 2, 8, stride=3, padding=8
                    ),
                    nn.ConvTranspose1d(
                        channel // 2, out_channel, 16, stride=4, padding=18
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose1d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


#  Make a VQVAE with merges like automouse_model
class VQVAE(Base):
    def __init__(
            self,
            in_channel=3,
            channel=128,
            n_res_block=2,
            n_res_channel=32,
            embed_dim=64,
            n_embed=512,
            dropout=0,
            decay=0.99
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.quantize_conv_t = nn.Conv1d(channel, embed_dim, 1)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv1d(channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose1d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dropout = nn.Dropout(p=dropout)
        self.dec = Decoder(
            embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        # self.random_init()

    def random_init(self, init_func=torch.nn.init.kaiming_uniform_):
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        quant_b, diff, _ = self.encode(input)
        # quant_t = self.dropout(quant_t)
        quant_b = self.dropout(quant_b).transpose(1, 2)
        dec = self.decode(quant_b)
        dec = torch.sigmoid(dec)
        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        # enc_t = self.enc_t(enc_b)

        # quant_t = self.quantize_conv_t(enc_t)  #.permute(0, 2, 1)
        # quant_t, diff_t, id_t = self.quantize_t(quant_t)
        # quant_t = quant_t  # .permute(0, 1, 2)
        # diff_t = diff_t.unsqueeze(0)

        # dec_t = self.dec_t(quant_t)
        # enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_b, diff_b, id_b

    def decode(self, quant_b):
        # upsample_t = self.upsample_t(quant_t)
        # quant = torch.cat([quant_b], 1)
        dec = self.dec(quant_b)

        return dec

    def decode_code(self, code_b):
        # quant_t = self.quantize_t.embed_code(code_t)
        # quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        # quant_b = quant_b  #.permute(0, 3, 1, 2)

        dec = self.decode(quant_b)

        return dec

    def get_model_name(self):
        return 'vqvae'