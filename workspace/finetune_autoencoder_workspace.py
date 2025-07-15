# Note: We need to import this before any package that might import
# transformer.
from common import transformers_pre_import_mods  # isort:skip

import datetime
import os
from typing import List, Tuple, Union

import hydra
import omegaconf
from omegaconf import OmegaConf
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy
from pytorch_lightning.utilities import rank_zero_only
import torch
import wandb

from common.path_util import resolve_path
from sgm.util import instantiate_from_config
from workspace.base_workspace import BaseWorkspace
from workspace.lightning_vae_callback import (
    SaveEveryNEpoch,
    SaveEveryNStep,
    LogPredictionSample,
)
from workspace.lightning_data_module import (
    DiffusionSpartanDataModule,
    DiffusionSpartanIterableDataModule,
)

### FSDP
from torch import nn
from torch.distributed.fsdp.wrap import *
import functools
import sgm

OmegaConf.register_new_resolver("eval", eval, replace=True)


def save_config(output_dir, config, basename):
    output_dir = resolve_path(output_dir)
    assert os.path.exists(output_dir)

    config_file = os.path.join(output_dir, f"{basename}.yaml")
    raw_yaml = OmegaConf.to_yaml(config)
    with open(config_file, "w") as file:
        print(raw_yaml, file=file)

    if wandb.run is not None:
        wandb.save(config_file, base_path=output_dir)


def maybe_configure_logging(resolved_cfg, global_rank_zero):
    """Configures logging settings and stores the resolved training configs
    only when `global_rank_zero=True`. This is to avoid saving multiple copies
    for the multi-node training scenario.
    """
    if global_rank_zero:
        output_dir = resolved_cfg.training["output_dir"]
        wandb_run = wandb.init(
            # sync_tensorboard=True,
            dir=str(output_dir),
            config=OmegaConf.to_container(resolved_cfg, resolve=True),
            **resolved_cfg.logging,
        )
        wandb.config.update(
            {
                "output_dir": output_dir,
            },
            # A SageMaker run fails without this flag as we
            # are overriding the `default` path from SageMaker. I couldn't
            # reproduce the issue locally at all.
            allow_val_change=True,
        )
        save_config(output_dir, resolved_cfg, basename="config")


class FinetuneAutoEncoderWorkspace(BaseWorkspace):
    """
    Implements LBM training using pytorch_lightning and facilitates checkpoint
    loading. Given a fully resolved config, this class is responsible for
    instantiating a PL module and a PL data module, configuring PL callbacks
    and logging, and setting up env variables for distributed training.

    Note that the config **should be resolved** prior to invoking this
    workspace. All the operations in the workspace and the classes within it
    should all refer to the resolved config as given without modifying it.
    """

    def __init__(self, resolved_cfg: OmegaConf):
        # Find a way to determine whether a OmegaConf is fully
        # resolved and error out if otherwise.
        assert isinstance(resolved_cfg, omegaconf.dictconfig.DictConfig)
        super().__init__(resolved_cfg)

        # We only want to save wandb logs for the main/first node during a
        # multinode setting. Thus, we need to get this information up front.
        self.global_rank_zero = self._check_global_rank_zero()

        # Class-wise setup.
        self._setup()

    def _setup(self):
        """Sets random seeds and CUDA/NCCL determinism, and instantiates
        necessary PL components, i.e., a PL module, a PL data module, and PL
        callbacks.
        """
        seed = self.resolved_cfg.training.seed
        seed_everything(seed, workers=True)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Create the output directory for checkpoints, logs, and configs.
        os.makedirs(self.output_dir, exist_ok=True)
        maybe_configure_logging(self.resolved_cfg, self.global_rank_zero)

        self.lightning_module_wrapper = self._create_lighting_module()
        self.lightning_data_module = self._create_lighting_data_module()
        self.lightning_callbacks = self._create_lighting_callbacks()

    def _check_global_rank_zero(self) -> bool:
        # This is some hack to get global rank before a trainer is
        # constructed. Normally you should do `trainer.global_rank` for this.
        # We should revisit the env variables for multinode
        # training. Torchrun, PyTorch DDP, Lightning seem to provide different
        # flavors, e.g., NODE_RANK vs. RANK.
        global_rank_zero = False
        if rank_zero_only.rank == 0:
            node_rank = os.environ.get("NODE_RANK", None)
            if node_rank is None or int(node_rank) == 0:
                # This is the main/first node.
                global_rank_zero = True
        return global_rank_zero

    def _create_lighting_module(self) -> LightningModule:
        """Configures models, i.e., both the observation encoders and the
        diffusion model, and returns a PL module.
        """
        lightning_module_wrapper = instantiate_from_config(self.resolved_cfg.model) # SVD DiffusionEngine
        return lightning_module_wrapper

    def _create_lighting_data_module(self) -> LightningDataModule:
        """Configures the data module for the training."""
        use_datapipe = (
            "build_spartan_datapipe" in self.resolved_cfg.task.dataset._target_
        )

        # Create a data module in npz or zarr format.
        if not use_datapipe:
            return DiffusionSpartanDataModule(self.resolved_cfg)

        # Create a data module of datapipe.
        if self.global_rank_zero:
            train_storage_dirs_config = [
                (
                    hydra.utils.instantiate(d)
                    if not isinstance(d, omegaconf.listconfig.ListConfig)
                    else d
                )
                for d in self.resolved_cfg.task.dataset.storage_dirs_list
            ]
            val_storage_dirs_config = [
                (
                    hydra.utils.instantiate(d)
                    if not isinstance(d, omegaconf.listconfig.ListConfig)
                    else d
                )
                for d in self.resolved_cfg.task.val_dataset.storage_dirs_list
            ]
            save_config(
                self.output_dir,
                train_storage_dirs_config,
                basename="train-datapipe",
            )
            save_config(
                self.output_dir,
                val_storage_dirs_config,
                basename="val-datapipe",
            )
        return DiffusionSpartanIterableDataModule(self.resolved_cfg)

    def _create_lighting_callbacks(self) -> List[ModelCheckpoint]:
        save_dir = os.path.join(self.output_dir, "checkpoints")
        lightning_callbacks = [
            # Save top k checkpoints.
            ModelCheckpoint(
                dirpath=save_dir,
                filename=self.resolved_cfg.checkpoint.topk.format_str,
                monitor=self.resolved_cfg.checkpoint.topk.monitor_key,
                mode=self.resolved_cfg.checkpoint.topk.mode,
                save_top_k=self.resolved_cfg.checkpoint.topk.k,
                verbose=True,
                save_last=self.resolved_cfg.checkpoint.save_last_ckpt,
                # Don't ever use this, as pl removes the previous ckpts even
                # for every_n_x.
                # every_n_epochs=self.resolved_cfg.checkpoint.every_n_epoch,
            ),
            LearningRateMonitor(),
        ]
        if self.resolved_cfg.checkpoint.get("every_n_epoch") is not None:
            lightning_callbacks.append(
                # Save every n epochs.
                SaveEveryNEpoch(
                    save_dir=save_dir,
                    period=self.resolved_cfg.checkpoint.every_n_epoch,
                    verbose=True,
                ),
            )
        if self.resolved_cfg.checkpoint.get("every_n_steps") is not None:
            lightning_callbacks.append(
                # Save every n steps.
                SaveEveryNStep(
                    save_dir=save_dir,
                    period=self.resolved_cfg.checkpoint.every_n_steps,
                    verbose=True,
                ),
            )
        if self.resolved_cfg.checkpoint.get("log_prediction_sample"):
            lightning_callbacks.append(
                LogPredictionSample(),
            )
        return lightning_callbacks

    def _get_distributed_training_settings(
        self,
    ) -> Tuple[int, int, Union[DDPStrategy, FSDPStrategy]]:
        """Extracts compute settings, i.e., the number of nodes and GPUs per
        node, and instantiates `Strategy` from the config. However, in the
        SageMaker case, we get these information from the runtime environment.
        """

        cfg_strategy = self.resolved_cfg.training.get("strategy", "ddp")
        print(f"Using Model Strategy: {cfg_strategy}")

        num_gpus = self.resolved_cfg.training.device
        num_nodes = self.resolved_cfg.training.get("num_nodes", 1)

        if cfg_strategy == "ddp":
            os.environ["NCCL_BLOCKING_WAIT"] = "1"

            strategy = DDPStrategy(
                timeout=datetime.timedelta(seconds=7200),
                find_unused_parameters=True,
            )
        elif cfg_strategy == "fsdp":
            policy = {sgm.modules.diffusionmodules.model.ResnetBlock,
                        sgm.modules.diffusionmodules.model.AttnBlock}
            strategy = FSDPStrategy(auto_wrap_policy=policy)
        else:
            raise Exception("Model strategy not supported!")

        return num_gpus, num_nodes, strategy


    def load_payload(self, payload, exclude_keys, include_keys, **kwargs):
        # Fix exclude_keys, and include_keys later.
        self.lightning_module_wrapper.load_state_dict(payload["state_dict"])

    def run(self, unit_test=False):
        assert (
            not self.resolved_cfg.training.train_with_features
        ), "Using `training.train_with_features=True` is probably broken now"

        # Prepare all the arguments for the pl Trainer.
        # Consider directly exposing the pl args for full
        # functionality.
        val_per_step = self.resolved_cfg.training.get("val_per_step", False)
        # Per https://lightning.ai/docs/pytorch/1.8.6/common/trainer.html.
        if val_per_step:
            # Turns off validataion at epoch end, and only val every N steps.
            val_check_interval = self.resolved_cfg.training.val_every
            check_val_every_n_epoch = None
        else:
            # Val only every N epochs.
            val_check_interval = None
            check_val_every_n_epoch = self.resolved_cfg.training.val_every

        num_gpus, num_nodes, strategy = (
            self._get_distributed_training_settings()
        )
        trainer = Trainer(
            devices=num_gpus,
            num_nodes=num_nodes,
            accelerator="gpu",
            strategy=strategy,
            max_epochs=self.resolved_cfg.training.num_epochs,
            max_steps=self.resolved_cfg.training.get("num_steps", -1),
            callbacks=self.lightning_callbacks,
            sync_batchnorm=True,
            precision=self.resolved_cfg.training.get("precision", 32),
            deterministic='warn',
            log_every_n_steps=1,
            check_val_every_n_epoch=check_val_every_n_epoch,
            val_check_interval=val_check_interval,
            limit_val_batches=self.resolved_cfg.training.get("max_val_steps", 1),
            logger=WandbLogger(save_dir=os.path.join(self.output_dir, "log")),
        )

        # For testing purposes, we only execute the code path up to this point,
        # i.e., without starting the actual training.
        if unit_test:
            return

        initial_weights = self.resolved_cfg.training.get(
            "initial_weights_from_checkpoint"
        )
        resume_checkpoint_path = self.resolved_cfg.training.get(
            "resume_checkpoint_path"
        )
        assert not (initial_weights and resume_checkpoint_path), (
            "Can't have resume_checkpoint_path and "
            "initial_weights_from_checkpoint specified at the same time."
        )
        if initial_weights:
            print(f"\nInitializing weights from {initial_weights}...\n")
            self.load_checkpoint(path=resolve_path(initial_weights))
        elif resume_checkpoint_path:
            print(f"\nResuming from: {resume_checkpoint_path}...\n")
            resume_checkpoint_path = resolve_path(resume_checkpoint_path)

        # If a checkpoint is specified, add it to the trainer args to resume.
        trainer.fit(
            model=self.lightning_module_wrapper,
            datamodule=self.lightning_data_module,
            ckpt_path=resume_checkpoint_path,
        )
