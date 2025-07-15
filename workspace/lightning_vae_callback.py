import os
import wandb
import numpy as np
import torch

from pytorch_lightning import Callback

from common.path_util import resolve_path

from imgaug.augmentables.heatmaps import HeatmapsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class PeriodicCheckpointCB(Callback):
    """
    Saves pl checkpoint periodically and when keyboard interrupted.
    The pl pytorch_lightning.callbacks.ModelCheckpoint doesn't really do
    periodic saving. It rather checks on some metric periodically, saves
    the best, and removes the old ones.
    """

    def __init__(
        self,
        period,
        save_dir,
        *,
        verbose=False,
        format="epoch_{epoch}-step_{step}.ckpt",
    ):
        self._period = period
        self._save_dir = save_dir
        self._verbose = verbose
        self._format = format

    def _save(self, trainer):

        # Evidently, this prevents appropriate synchronization
        # across ranks in DDP for PL version > 2.0 resulting in deadlocks(?).
        # See: https://github.com/Lightning-AI/pytorch-lightning/issues/19045
        # if rank_zero_only.rank != 0:
        #     return

        epoch = trainer.current_epoch
        step = trainer.global_step
        relpath = self._format.format(epoch=epoch, step=step)
        ckpt_path = os.path.join(self._save_dir, relpath)

        # Only make the checkpoint directory for global rank 0 since that is
        # the only rank that will actually save a checkpoint. The other ranks
        # will have different ckpt_path and create unnecessary unused dirs.
        if trainer.global_rank == 0:
            os.makedirs(self._save_dir, exist_ok=True)

        # This should do nothing for trainer.global_rank != 0, but needs to be
        # called for all rank to enable proper syncing (apparently).
        trainer.save_checkpoint(ckpt_path)
        if self._verbose and trainer.global_rank == 0:
            print(f"  {type(self)}: Saved to {ckpt_path}")


class SaveEveryNEpoch(PeriodicCheckpointCB):
    """
    Since PL's training loop calls this at the end of the validation step,
    using this callback can result in more than intended number of checkpoints
    being saved. For example, if you configure `val_check_interval` to be less
    than number of training batches per epoch, this callback will be called
    multiple times per epoch.
    """

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        step = trainer.global_step

        if (
            not isinstance(self._period, str)
            and step != 0
            and (epoch % self._period == 0)
        ) and (step > 20000):
            self._save(trainer)


class SaveEveryNStep(PeriodicCheckpointCB):
    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        step = trainer.global_step
        if (
            step != 0
            and not isinstance(self._period, str)
            and step % self._period == 0
        ):
            self._save(trainer)

def unnormalize(input, min, max):
    return torch.clamp(((input + 1.) / 2.) * (max - min) + min, min, max)

class LogPredictionSample(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        
        N, C, H, W = outputs['reconstructions'].shape
        
        gt_colors = torch.tile(torch.tensor([0., 255., 0.]), (H * W, 1))
        val_colors = torch.tile(torch.tensor([255., 0., 0.]), (H * W, 1))
        recon_colors = torch.tile(torch.tensor([0., 0., 255.]), (H * W, 1))
        
        # (n_frames, C, H, W)
        gt_pointmap = ((outputs['inputs'][0])).permute(1, 2, 0).reshape(-1, 3).cpu().detach()
        gt_pointmap_colored = torch.cat([gt_pointmap, gt_colors], dim=-1)
        
        recon_pointmap = (outputs['reconstructions'][0]).permute(1, 2, 0).reshape(-1, 3).cpu().detach()
        recon_pointmap_colored = torch.cat([recon_pointmap, recon_colors], dim=-1)
        
        if wandb.run is not None:
            recon = torch.cat([recon_pointmap_colored, gt_pointmap_colored], dim=0)
            wandb.log({"val/gt_pointmap": wandb.Object3D(np.array(gt_pointmap_colored))})
            wandb.log({"val/recon_pointmap": wandb.Object3D(np.array(recon_pointmap_colored))})
            wandb.log({"val/recon_pointmap_overlay": wandb.Object3D(np.array(recon))})
            wandb.log({"val/recon_pointmap_error": torch.mean(torch.square(gt_pointmap - recon_pointmap))})

        # gt_images = unnormalize(outputs['inputs'][:N//2], 0, 255).cpu().detach()
        # recon_images = unnormalize(outputs['reconstructions'][:N//2], 0, 255).cpu().detach()
        
        # if wandb.run is not None:
        #     wandb.log({"val/gt_images": wandb.Video(np.array(gt_images, dtype=np.uint8))})
        #     wandb.log({"val/recon_images": wandb.Video(np.array(recon_images, dtype=np.uint8))})
        #     wandb.log({"val/recon_images_error": torch.mean(torch.square(gt_images - recon_images))})
