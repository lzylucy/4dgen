import os
from os.path import expanduser, expandvars
import pathlib

import hydra
from pytorch_lightning import LightningDataModule
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torchdata.dataloader2 import (
    DataLoader2,
    DistributedReadingService,
    MultiProcessingReadingService,
    SequentialReadingService,
)

from common.normalize_util import (
    get_linear_normalizer_from_saved_stats,
)
from common.path_util import (
    resolve_glob_type_to_list,
    resolve_path,
)
from model.common.normalizer import (
    SingleFieldLinearNormalizer,
    create_normalizer_with_identity,
)


class DiffusionSpartanIterableDataModule(LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self._config = cfg

    def setup(self, stage: str):
        print(f"[{__class__.__name__}] Instantiating datapipes")

        self._train_datapipe = hydra.utils.instantiate(
            self._config.task.dataset,
        )
        self._val_datapipe = hydra.utils.instantiate(
            self._config.task.val_dataset,
        )

        # Wait for cache files to be downloaded on rank_zero process.
        dist.barrier()

        shape_meta = self._config.task.shape_meta

        normalizer_path = self._config.task.dataset.get(
            "normalizer_path", None
        )
        if normalizer_path is None:
            self.normalizer = create_normalizer_with_identity(shape_meta)
        else:
            normalizer = get_linear_normalizer_from_saved_stats(
                resolve_path(normalizer_path)
            )

            camera_names = []
            for k, attr in shape_meta.obs.items():
                if attr.type == "rgb":
                    camera_names.append(k)
            # image
            for camera_name in camera_names:
                normalizer[camera_name] = (
                    SingleFieldLinearNormalizer.create_identity()
                )
            self.normalizer = normalizer

    def prepare_data(self):
        pass

    def train_dataloader(self):
        rs = self._get_reading_service(self._config.dataloader.num_workers)
        # We need to keep a reference of the object to shut it down in the end.
        self._train_dataloader = DataLoader2(
            datapipe=self._train_datapipe,
            reading_service=rs,
        )
        return self._train_dataloader

    def val_dataloader(self):
        rs = self._get_reading_service(self._config.val_dataloader.num_workers)
        # We need to keep a reference of the object to shut it down in the end.
        self._val_dataloader = DataLoader2(
            datapipe=self._val_datapipe,
            reading_service=rs,
        )
        return self._val_dataloader

    def _get_reading_service(self, num_workers: int = 1):
        rs = MultiProcessingReadingService(
            num_workers=num_workers,
            # Adding prefetch at the end of each worker doesn't help at all so
            # desabling it.
            worker_prefetch_cnt=0,
        )

        if (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() != 1
        ):
            print(
                f"[{__class__.__name__}] Distributed and MultiProcessing "
                f"ReadingService with num_workers: {num_workers}"
            )
            rs = SequentialReadingService(DistributedReadingService(), rs)
        else:
            print(
                f"[{__class__.__name__}] MultiProcessingReadingService with "
                f"num_workers: {num_workers}"
            )
        return rs

    def teardown(self, stage: str):
        # Shutdown DataLoader2 to avoid hanging behavior when training is
        # finished.
        self._train_dataloader.shutdown()
        self._val_dataloader.shutdown()


class DiffusionSpartanDataModule(LightningDataModule):
    def __init__(self, full_cfg):
        """
        pl data module wrapper around spartan dataset
        """
        super().__init__()
        self._cfg = full_cfg

        # this only matters for zarr backend replay buffer.
        # - if s3, will download from this s3 link to
        #       ~/tmp/s3_link_dir, and then load from there.
        # - if local path, will directly load zarr replay buffer from there
        #       if there exists a zarr replay buffer. if a zarr isn't present
        #       will build one from spartan and save to local path.
        self.replay_buffer_path = self._cfg.task.dataset.get(
            "replay_buffer_path", None
        )

        self.prepare_data_per_node = True

    def prepare_data(self):
        """
        This is called for each nodes' rank 0 process.
        The intention is to prepare the dataset (replay buffer) as much as
        possible ahead of training time.
        - If the replay buffer backend is set to np, this does nothing
        - If no desired on disk replay buffer storage
            (cfg.task.dataset.replay_buffer_path) is supplied, this does
            nothing.
        - If a s3 link is supplied as on disk replay buffer storage, this will
            download from that s3 link to a local cache if necessary.
            If the local storage is present already, this does nothing.
        - If a local file system path is given but no replay buffer is there,
            this will load from all spartan data, build the replay buffer, and
            write to the specified local storage location.
        - If a local file system path is given, and a replay buffer is there,
            this does nothing.
        """
        if self.replay_buffer_path is None:
            return

        if self._cfg.task.dataset.mode == "np":
            print(
                "\n\nWARNING: np based replay buffer doesn't support on disk "
                "storage\n\n"
            )
            return

        # handles local fs path
        has_dataset = os.path.exists(
            expandvars(expanduser(self.replay_buffer_path))
        )
        if has_dataset:
            # don't need to do anything here.
            return

        print("\n\nMaking a on-disk zarr dataset from:")
        for eps in self._cfg.task.dataset.episode_path_globs:
            print(f"  {eps}")

        dataset = hydra.utils.instantiate(self._cfg.task.dataset)
        replay_buffer = dataset.replay_buffer

        replay_buffer.save_to_path(
            zarr_path=self.replay_buffer_path,
        )
        print(f"  Saved replay buffer to {self.replay_buffer_path}")

    def setup(self, stage: str):
        """
        This is run for each process on each node.
        - For np based replay buffer or zarr bazed but no on disk cache, this
            will always build a dataset from spartan.
        - For zarr based with on disk cahce, prepare_data() should ensure that
            there is a local copy at self.replay_buffer_path, so we will build
            a replay buffer from there.
        """

        # prepare should ensure there is a local zarr dataset built at
        # self.replay_buffer_path. for np based dataset, this should have no
        # effects.
        self._cfg.task.dataset.replay_buffer_path = self.replay_buffer_path
        self._train_dataset = hydra.utils.instantiate(self._cfg.task.dataset)
        self._val_dataset = self._train_dataset.get_validation_dataset()

        # sanity check that the episode paths match whatever in the glob
        # string
        if self._train_dataset.replay_buffer.episode_paths is not None:
            # episode_paths isn't available for np based replay_buffer, and
            # can be missing from legacy on disk zarr replay buffer.
            resolved_globs = resolve_glob_type_to_list(
                list(self._cfg.task.dataset.episode_path_globs)
            )
            stored_paths = set(self._train_dataset.replay_buffer.episode_paths)
            for path in resolved_globs:
                assert path in stored_paths, f"{path} not in zarr dataset"
            assert len(stored_paths) == len(resolved_globs)

        self.normalizer = self._train_dataset.get_normalizer()

    def train_dataloader(self):
        return DataLoader(self._train_dataset, **self._cfg.dataloader)

    def val_dataloader(self):
        return DataLoader(self._val_dataset, **self._cfg.val_dataloader)
