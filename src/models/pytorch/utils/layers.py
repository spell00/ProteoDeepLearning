from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel, in_channel, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x

        return out


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

    def forward(self, x):
        return self.blocks(x)
