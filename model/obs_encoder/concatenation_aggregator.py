from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class ConcatenationAggregator(nn.Module):
    """
    in: dict of tensors
    out: a single tensor
    Tensor concatenated in the order specified by in_keys
    """

    def __init__(
        self,
        # defined the order
        in_keys: List[str],
        keep_first_n_dims: int = 1,
    ):
        super().__init__()
        self.in_keys = in_keys
        self.keep_first_n_dims = keep_first_n_dims

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = list()
        for key in self.in_keys:
            value = x[key]
            feature = value.reshape(*value.shape[: self.keep_first_n_dims], -1)
            features.append(feature)
        feature = torch.cat(features, dim=-1)
        return feature


def test():
    cat_agg = ConcatenationAggregator(
        ["img", "agent_pos"], keep_first_n_dims=2
    )
    x = {"img": torch.zeros((1, 8, 256)), "agent_pos": torch.zeros((1, 8, 2))}
    o = cat_agg(x)
