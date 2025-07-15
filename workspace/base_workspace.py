import copy
import os
import pathlib
import threading
from typing import Optional

import dill
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch

from common.path_util import resolve_path


class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, resolved_cfg: OmegaConf):
        # `output_dir` needs to be specified either from the yaml config or a
        # terminal argument which will be resolved by now.
        output_dir = resolved_cfg.training.get("output_dir")
        assert output_dir is not None, "`output_dir` is not specified."
        self.output_dir = resolve_path(output_dir)

        self.resolved_cfg = resolved_cfg
        self._saving_thread = None

    def run(self):
        """
        Create any resource shouldn't be serialized as local variables
        """
        pass

    def save_checkpoint(
        self,
        path=None,
        tag="latest",
        exclude_keys=None,
        include_keys=None,
        use_thread=True,
    ):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath(
                "checkpoints", f"{tag}.ckpt"
            )
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {"cfg": self.cfg, "state_dicts": dict(), "pickles": dict()}

        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(
                value, "load_state_dict"
            ):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload["state_dicts"][key] = _copy_to_cpu(
                            value.state_dict()
                        )
                    else:
                        payload["state_dicts"][key] = value.state_dict()
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(
                    payload, path.open("wb"), pickle_module=dill
                )
            )
            self._saving_thread.start()
        else:
            torch.save(payload, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    def get_checkpoint_path(self, tag="latest"):
        return pathlib.Path(self.output_dir).joinpath(
            "checkpoints", f"{tag}.ckpt"
        )

    def load_checkpoint(
        self,
        path=None,
        tag="latest",
        exclude_keys=None,
        include_keys=None,
        **kwargs,
    ):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open("rb"), pickle_module=dill, **kwargs)
        self.load_payload(
            payload, exclude_keys=exclude_keys, include_keys=include_keys
        )
        return payload

    def load_payload(
        self, payload, exclude_keys=None, include_keys=None, **kwargs
    ):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload["pickles"].keys()

        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    @classmethod
    def create_from_checkpoint(
        cls,
        *,
        checkpoint_path,
        output_dir,
        exclude_keys=None,
        include_keys=None,
        **kwargs,
    ):
        assert (
            output_dir is not None
        ), "output_dir needs to be spcified during workspace creation."
        payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill)

        # These are some backward compatibility hacks to load old dp
        # checkpoints.
        converting = False
        new_state_dict = {}
        for k, v in payload["state_dict"].items():
            if k.startswith("ema_model"):
                name_afterwards = k.partition("ema_model")[2]
                new_state_dict[f"ema.averaged_model{name_afterwards}"] = v
                converting = True
            else:
                new_state_dict[k] = v
        if converting:
            print("======================================================")
            print("YOU ARE LOADING A LEGACY CHECKPOINT WITH OLD EMA MODEL")
            print("  changing ema_model.X to ema.model.X in state_dict")
            print("  adding ema.optimization_step = 0 to state_dict")
            print("======================================================")
            # this wasn't saved in the old checkpoints, but should have.
            new_state_dict["ema.optimization_step"] = torch.zeros(
                1, dtype=torch.long
            )
            payload["state_dict"] = new_state_dict

        ema_config = payload["cfg"].get("ema", {})
        ema_target = ema_config.get("_target_", "")
        if (
            ema_target
            == "diffusion_policy.model.diffusion.ema_model.TorchEMAModel"
        ):
            print("======================================================")
            print("YOU ARE LOADING A LEGACY CHECKPOINT WITH OLD EMA MODEL")
            print("  renaming EMA module name in cfg.")
            print("======================================================")
            ema_config["_target_"] = (
                "diffusion_policy.model.diffusion.ema_model.VanillaEMAModel"
            )

        # Note: Reusing `output_dir` from the checkpoint file is never ideal
        # due to different users and machines. Overwrite it before
        # instantiating the class.
        payload["cfg"]["training"]["output_dir"] = output_dir
        instance = cls(payload["cfg"])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs,
        )
        return instance

    def save_snapshot(self, tag="latest"):
        """
        Quick loading and saving for reserach, saves full state of the
        workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath(
            "snapshots", f"{tag}.pkl"
        )
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, "rb"), pickle_module=dill)


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)
