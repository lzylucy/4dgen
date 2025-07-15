import torch
import torch.nn as nn

from model.diffusion.conv1d_components import (
    Conv1dBlock,
    Downsample1d,
)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, n_groups=8):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                Conv1dBlock(
                    in_channels, out_channels, kernel_size, n_groups=n_groups
                ),
                Conv1dBlock(
                    out_channels, out_channels, kernel_size, n_groups=n_groups
                ),
            ]
        )

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        """
        x : [ batch_size x inp_channels x horizon ]
        t : [ batch_size x embed_dim ]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks(x)
        return out + self.residual_conv(x)


class TemporalAggregator(nn.Module):
    def __init__(
        self,
        in_channels,
        channel_mults=(2,),
        n_blocks_per_level=1,
        kernel_size=5,
        n_groups=8,
    ):
        super().__init__()
        channels = [in_channels * x for x in channel_mults]
        blocks = list()
        ic = in_channels
        for i, oc in enumerate(channels):
            level = list()
            for _ in range(n_blocks_per_level):
                level.append(
                    ResidualTemporalBlock(
                        ic, oc, kernel_size=kernel_size, n_groups=n_groups
                    )
                )
                ic = oc
            level.append(Downsample1d(oc))
            blocks.append(nn.Sequential(*level))
            ic = oc
        # final conv
        blocks.append(nn.Conv1d(ic, ic, 1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        """
        in: (B,T,D)
        out: (B,D')
        """
        x = torch.moveaxis(x, 1, 2)
        # (B,D,T)
        x = self.blocks(x)
        # average pooling
        x = x.mean(dim=-1)
        # (B,D')
        return x


def test():
    tagg = TemporalAggregator(in_channels=256)
    x = torch.zeros((1, 16, 256))
    o = tagg(x)
    # print(o.shape)
