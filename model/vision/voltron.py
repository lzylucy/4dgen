import os

from hydra.utils import to_absolute_path
import torch.nn as nn
from voltron import instantiate_extractor, load

class VoltronVisual(nn.Module):
    def __init__(self, weights="v-cond", frozen=False, inference_text=None):

        super().__init__()

        # `/opt/ml/code/cache` is our unified directory to put cached weights
        # when running on SageMaker. Otherwise, we will put them in `/tmp`.
        cache = "/opt/ml/code/cache" if is_sagemaker() else "/tmp"
        self.vcond, self.preprocess = load(
            weights, device="cpu", freeze=frozen, cache=cache
        )
        self.ve = instantiate_extractor(self.vcond)()

    def forward(self, x, lang=None):

        # For now, assert correct shape. Also consider using
        # self.preprocess(x).
        assert x.shape[1] == 3 and x.shape[2] == 224 and x.shape[3] == 224

        # Call the forward pass.
        if lang is None:
            output = self.vcond.get_representations(x, mode="visual")
        else:
            tokens = lang["input_ids"].to(self.vcond.lm.device)
            token_mask = lang["attention_mask"].to(self.vcond.lm.device)
            output = self.vcond.encode(x, tokens, token_mask)

        final_rep = self.ve(output)
        return final_rep
