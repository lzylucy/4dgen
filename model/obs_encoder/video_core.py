from functools import partial
from typing import Optional

from pytorchvideo import models
import torch
import torch.nn as nn


class VideoResNet(nn.Module):
    def __init__(self, norm_groups=None, **kwargs):
        super().__init__()
        if norm_groups is not None:
            if norm_groups == 0:
                norm_layer = nn.Identity
            else:
                norm_layer = lambda num_features, eps, **kwargs: nn.GroupNorm(
                    num_groups=norm_groups, num_channels=num_features, eps=eps
                )
            kwargs["norm"] = norm_layer
        backbone = models.create_resnet(**kwargs)
        self.net = nn.Sequential(*backbone.blocks[:5])

    def forward(self, x):
        return self.net(x)


class VideoCore(nn.Module):
    def __init__(self, backbone: nn.Module, pool: Optional[nn.Module] = None):
        """
        backbone, nn.Module:
            in: (B,T,C,H,W)
            out: (B,D,t,h,w)

        pool, nn.Module:
            in: (B,D,t,h,w)
            out: (B,D')
        """

        super().__init__()
        self.backbone = backbone
        self.pool = pool

    def output_shape(self, input_shape):
        output_shape = self.backbone.output_shape(input_shape)
        if self.pool is not None:
            output_shape = self.pool.output_shape(output_shape)
        return output_shape

    def forward(self, x):
        x = torch.moveaxis(x, 1, 2)
        # (B,C,T,H,W)
        x = self.backbone(x)
        if self.pool is not None:
            x = self.pool(x)
        return x
