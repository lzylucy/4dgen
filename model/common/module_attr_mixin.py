import torch
import torch.nn as nn


class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        # We initialize a variable and set it to not have a gradient as this
        # allows the state optimizer dict to work properly with FSDP.
        # Removing it altogether causes the policy to not learn.
        self._dummy_variable = nn.Parameter(
            torch.tensor(
                [0.0],
                requires_grad=False,
            )
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
