from typing import Dict, List

import torch
import torch.nn as nn


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        obs_type_keys: Dict[str, List[str]],
        obs_type_nets: Dict[str, nn.Module],
    ):
        pass
