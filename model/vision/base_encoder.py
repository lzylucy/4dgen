from typing import Dict

import torch

from model.common.module_attr_mixin import ModuleAttrMixin


class BaseEncoder(ModuleAttrMixin):
    def forward(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Predict the encoding associated with a set of observation inputs.

        Each subclass of BaseEncoder must implement this method. In typical
        usage, the input obs_dict will contain observation data (e.g.,
        camera images) and will output a dict containing an encoding.

        Args:
            obs_dict: Dict containing observation data.

        Returns:
            Dict containing one or more tensors of encodings.

        """
        raise NotImplementedError()
