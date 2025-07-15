"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import pathlib

import hydra
from omegaconf import OmegaConf

from common.debug_util import iex
from workspace.base_workspace import BaseWorkspace

# Allows arbitrary python code execution in configs using the ${eval:''}
# resolver.
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config"))
)
@iex
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
