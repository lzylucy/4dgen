import copy
import functools
import os
from typing import Dict, List, Optional

import cv2
from einops import rearrange
import numcodecs
import numpy as np
from omegaconf.listconfig import ListConfig
import torch
from tqdm import tqdm
import yaml

from my_codecs.imagecodecs_numcodecs import (
    Jpeg2k,
    register_codecs,
)
from common.fsspec_util import (
    load_file_with_fsspec,
    load_npz_with_fsspec,
)
from common.normalize_util import (
    array_to_stats,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_range_normalizer_from_stat,
)
from common.parallel_work import parallel_work
from common.path_util import resolve_glob_type_to_list
from common.pytorch_util import dict_apply, dict_apply_reduce
from common.replay_buffer import ReplayBuffer
from common.sampler import SequenceSampler, get_val_mask
from dataset.base_dataset import BaseImageDataset
from dataset.relative_trajectory_conversion import (
    change_to_relative_trajectories,
)
from model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from model.common.rotation_transformer import (
    get_rotation_transformer,
)

register_codecs()


class SpartanBaseDataset(BaseImageDataset):
    def __init__(
        self,
        episode_path_globs: str,
        shape_meta: dict,
        imagenet_normalization: bool,
        horizon=1,
        n_obs_steps=1,
        pad_before=0,
        pad_after=0,
        repeat_head=0,
        repeat_tail=0,
        stride=1,
        has_gripper=False,
        rotation_rep="rotation_6d",
        val_ratio=0.0,
        seed=42,
        mode="np",
        compressor="blosc",
        max_num_episodes: Optional[int] = None,
        raw_rgb=True,
        num_workers=8,
        replay_buffer_path=None,
        is_multiarm=False,
        is_relative=False,
        path_is_fully_resolved: bool = False,
        has_depth: bool = False,
        has_label: bool = False,
    ):
        if is_relative:
            assert (
                is_multiarm
            ), "Relative trajectories not supported for single arm data."
        # hack..
        assert rotation_rep in ["rotation_6d", "pitch"]

        obs_shape_meta = shape_meta["obs"]

        self._is_multiarm = is_multiarm
        self._camera_names = []
        self._lowdim_names = []
        self._raw_rgb = raw_rgb
        self._shape_meta = shape_meta
        self._is_relative = is_relative

        print(f"\n\nIs relative traj: {self._is_relative}\n")
        # fail fast check
        for arm in ["left", "right"]:
            other_arm = "right" if arm == "left" else "left"
            if self._is_relative:
                assert (
                    f"robot__actual__poses__{arm}__{other_arm}::panda__xyz"
                    in obs_shape_meta
                )
                assert (
                    f"robot__actual__poses__{arm}__{other_arm}::panda__rot_6d"
                    in obs_shape_meta
                )
            else:
                assert (
                    f"robot__actual__poses__{arm}__{other_arm}::panda__xyz"
                    not in obs_shape_meta
                )
                assert (
                    f"robot__actual__poses__{arm}__{other_arm}::panda__rot_6d"
                    not in obs_shape_meta
                )

        # camera_name: (h, w)
        self._image_shapes = {}
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])

            # type defaults to "low_dim"
            type = attr.get("type", "low_dim")

            if type == "rgb":
                self._camera_names.append(key)
                assert len(shape) == 3

                self._image_shapes[key] = shape[1:]

            elif type == "low_dim":
                self._lowdim_names.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # idk how to deal with this correctly
        if isinstance(episode_path_globs, ListConfig):
            episode_path_globs = list(episode_path_globs)

        if path_is_fully_resolved:
            episode_paths = episode_path_globs
        else:
            episode_paths = resolve_glob_type_to_list(episode_path_globs)

        if max_num_episodes is not None and (
            max_num_episodes < len(episode_paths)
        ):
            episode_paths = episode_paths[:max_num_episodes]
            assert len(episode_paths) == max_num_episodes

        # check if replay buffer already exists
        if replay_buffer_path is None:
            # load all episodes
            print(f"Building replay buffer from scratch with stride: {stride}")
            replay_buffer = make_replay_buffer(
                episode_paths=episode_paths,
                rotation_rep=rotation_rep,
                repeat_head=repeat_head,
                repeat_tail=repeat_tail,
                mode=mode,
                compressor=compressor,
                num_workers=num_workers,
                stride=stride,
                is_multiarm=is_multiarm,
                camera_names=self._camera_names,
                lowdim_names=self._lowdim_names,
                image_shapes=self._image_shapes,
                has_gripper=has_gripper,
                has_depth=has_depth,
                has_label=has_label,
            )
            # print(f"Save replay buffer to {replay_buffer_path}")
            # replay_buffer.save_to_path(replay_buffer_path)
        else:
            print(f"Loading replay buffer from: {replay_buffer_path}")
            if replay_buffer_path.startswith("s3://"):
                replay_buffer = ReplayBuffer.create_from_s3(replay_buffer_path)
            else:
                replay_buffer = ReplayBuffer.create_from_path(
                    replay_buffer_path
                )

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.replay_buffer = replay_buffer
        self.sampler = sampler

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()


class EpisodePathHelper:
    def __init__(self, episode_path: str) -> None:
        self.episode_path = episode_path
        self._base_path = os.path.join(self.episode_path, "processed")

    @property
    def actions_path(self) -> str:
        path = os.path.join(self._base_path, "actions.npz")
        assert os.path.exists(path), f"{path} doesn't exist."
        return path

    @property
    def observations_path(self) -> str:
        path = os.path.join(self._base_path, "observations.npz")
        assert os.path.exists(path), f"{path} doesn't exist."
        return path

    @property
    def metadata_path(self) -> str:
        path = os.path.join(self._base_path, "metadata.yaml")
        assert os.path.exists(path), f"{path} doesn't exist."
        return path
    
    @property
    def cam_extrs_path(self) -> str:
        path = os.path.join(self._base_path, "extrinsics.npz")
        assert os.path.exists(path), f"{path} doesn't exist."
        return path
    
    @property
    def cam_intrs_path(self) -> str:
        path = os.path.join(self._base_path, "intrinsics.npz")
        assert os.path.exists(path), f"{path} doesn't exist."
        return path


def make_replay_buffer(
    *,
    episode_paths: List[str],
    rotation_rep: str,
    repeat_head: int,
    repeat_tail: int,
    mode: str,
    compressor: str,
    num_workers: int,
    stride: int,
    is_multiarm: bool,
    camera_names: List[str],
    lowdim_names: List[str],
    image_shapes: Dict[str, tuple],
    has_gripper: Optional[bool],
    has_depth: Optional[bool],
    has_label: Optional[bool],
):
    assert mode in ["np", "zarr"], f"Unknown storage mode: {mode}"
    assert compressor in [
        "blosc",
        "jpeg2k",
    ], f"Unknown compressor: {compressor}"

    if mode == "np":
        replay_buffer = ReplayBuffer.create_empty_numpy()
    elif mode == "zarr":
        replay_buffer = ReplayBuffer.create_empty_zarr()

    if compressor == "blosc":
        compressor = numcodecs.Blosc(
            cname="lz4", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE
        )
    elif compressor == "jpeg2k":
        compressor = Jpeg2k(level=50)

    if not is_multiarm:
        rotation_transformer = get_rotation_transformer(
            from_rep="axis_angle",
            to_rep=rotation_rep,
        )

    def worker(paths):
        for episode_path in paths:
            # load the obj and action trajs.
            path_helper = EpisodePathHelper(episode_path)

            observations = load_npz_with_fsspec(
                path_helper.observations_path,
            )
            actions = load_npz_with_fsspec(
                path_helper.actions_path,
            )
            metadata = load_file_with_fsspec(
                path_helper.metadata_path,
                yaml.safe_load,
            )
            cam_extrs = load_npz_with_fsspec(
                path_helper.cam_extrs_path,
                yaml.safe_load,
            )
            cam_intrs = load_npz_with_fsspec(
                path_helper.cam_intrs_path,
                yaml.safe_load,
            )
            camera_dict = dict()
            camera_dict['intr'] = cam_intrs
            camera_dict['extr'] = cam_extrs

            assert (
                "camera_id_to_semantic_name" in metadata
            ), f"Missing camera_id_to_semantic_name in {episode_path}"
            camera_id_to_camera_name = metadata["camera_id_to_semantic_name"]

            assert len(metadata["skills"].items()) == 1
            skill_name = next(iter(metadata["skills"].keys()))

            if is_multiarm:
                episode = bimanual_spartan_episode_to_replay_buffer_episode(
                    observation_dict=observations,
                    camera_dict=camera_dict,
                    action_dict=actions,
                    skill_name=skill_name,
                    lowdim_keys=lowdim_names,
                    camera_names=camera_names,
                    camera_id_to_camera_name=camera_id_to_camera_name,
                    repeat_head=repeat_head,
                    repeat_tail=repeat_tail,
                    image_shapes=image_shapes,
                    has_depth=has_depth,
                    has_label=has_label,
                )
            else:
                episode = spartan_episode_to_replay_buffer_episode(
                    observation_dict=observations,
                    action_dict=actions,
                    skill_name=skill_name,
                    lowdim_keys=lowdim_names,
                    camera_names=camera_names,
                    camera_id_to_camera_name=camera_id_to_camera_name,
                    rotation_transformer=rotation_transformer,
                    repeat_head=repeat_head,
                    repeat_tail=repeat_tail,
                    has_gripper=has_gripper,
                    image_shapes=image_shapes,
                )
            # we take stride > 1 to downsample in the time dimension.
            # this only takes effect when making a new replay buffer from
            # spartan data.
            if stride != 1:
                strided = {}
                for k, v in episode.items():
                    # T should be the first dim.
                    index = np.arange(0, v.shape[0], step=stride)
                    strided[k] = v[index]
                episode = strided

            yield (episode, episode_path)

    episodes = parallel_work(
        worker,
        episode_paths,
        process_count=num_workers,
        progress_cls=functools.partial(
            tqdm, desc=f"Loading episodes w. {num_workers} proc"
        ),
    )

    for episode, path in tqdm(episodes, desc="Adding to ReplayBuffer"):
        # add image compressors, we will the vector ones.
        compressors = {}
        chunks = {}
        for cam_name in camera_names:
            name = f"obs.{cam_name}"
            compressors[name] = compressor
            _, h, w, c = episode[name].shape
            chunks[name] = (1, h, w, c)
        replay_buffer.add_episode(
            episode,
            path=path,
            # compressors arg have no effect for np backend.
            compressors=compressors,
            chunks=chunks,
        )

    return replay_buffer


def get_normalizer_params(x, output_max=1, output_min=-1, range_eps=1e-7):
    N, d = x.shape
    # 4 is for pitch
    # assert d in [9, 10, 1, 4]
    # assume always [xyz, 6d_rot, maybe_gripper]
    # assume always [xyz, 1d_pitch, maybe_gripper]
    input_min = np.min(x, axis=0)
    input_max = np.max(x, axis=0)
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps

    # set rotation dims to always ignore, since these are always in -1 to 1
    # if d != 1:
    #    ignore_dim[3:9] = True

    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    # scale[ignore_dim] = 1
    offset = output_min - scale * input_min
    # offset[ignore_dim] = (
    #     (output_max + output_min) / 2 - input_min[ignore_dim]
    # )
    offset[ignore_dim] = 0

    info = {
        "min": input_min,
        "max": input_max,
        "mean": np.mean(x, axis=0),
        "std": np.std(x, axis=0),
    }

    return scale, offset, info


def _change_pose_repr(poses, rotation_transformer):
    N, d = poses.shape
    assert N >= 1

    # assuming gripper width...
    if d == 1:
        return poses.astype(np.float32)

    assert d == 6
    # assuming xyz, axang
    rot = poses[:, 3:]
    pos = poses[:, :3]
    new_rot = rotation_transformer.forward(rot)

    result = np.concatenate([pos, new_rot], axis=-1).astype(np.float32)
    return result


def _repeat(x, num, mode="head"):
    assert num >= 0
    if mode == "head":
        stuff = [x[:1] for i in range(num)]
    else:
        stuff = [x[-1:] for i in range(num)]
    if mode == "head":
        return np.concatenate(stuff + [x], axis=0)
    else:
        return np.concatenate([x] + stuff, axis=0)
    
def rgb2gray(imgs):
    return np.dot(imgs[...,:3], [0.2989, 0.5870, 0.1140])


def depth_imgs_to_maps(depth_images, cam_intr):
    pointmap = np.zeros((len(depth_images), depth_images.shape[1], depth_images.shape[2], 3))
    for v in range(depth_images.shape[1]):
        for u in range(depth_images.shape[2]):
            Z = depth_images[:, v, u]
            # if Z == 0:
            #     continue
            X = (u - cam_intr[0, 2]) * Z / cam_intr[0, 0]
            Y = (v - cam_intr[1, 2]) * Z / cam_intr[1, 1]
            pointmap[:, v, u] = np.concatenate([X[:, None], Y[:, None], Z[:, None]], axis=-1)
    
    pointmap = pointmap / 1000.
    return pointmap


def apply_transform_on_depthmaps(dm, t):
    B, H, W, C = dm.shape
    ps = dm.reshape(B, -1, C)
    ps_homogeneous = torch.cat([ps, torch.ones((B, H*W, 1))], dim=-1)
    ps_transformed = torch.transpose(torch.matmul(t, torch.transpose(ps_homogeneous, 1, 2)), 1, 2)
    ps_transformed = ps_transformed[:, :, :C]
    dm_transformed = ps_transformed.reshape(B, H, W, C)

    return dm_transformed


def _add_spartan_episode_images_to_replay_buffer_episode(
    *,
    observation_dict,
    camera_names: List[str],
    camera_id_to_camera_name: Dict[str, str],
    repeat_head: int,
    repeat_tail: int,
    image_shapes: Dict[str, tuple],
    has_depth: bool,
    has_label: bool,
):
    data = {}
    # Loop through `observation_dict` and find cameras.
    camera_names_found = []
    for observation_key in observation_dict.keys():
        if observation_key not in camera_names:
            continue

        # camera_name = camera_id_to_camera_name[observation_key]
        # camera_names_found.append(camera_name)

        # if camera_name not in image_shapes:
        #     continue

        camera_name = observation_key
        imgs = observation_dict[observation_key]
        imgs = _repeat(imgs, num=repeat_head, mode="head")
        # (T, H, W, 3) unnormalized.
        imgs = _repeat(imgs, num=repeat_tail, mode="tail")
        T, H, W, C = imgs.shape

        # resize
        resized = []
        h, w = image_shapes[camera_name]
        for i in range(imgs.shape[0]):
            resized.append(
                cv2.resize(imgs[i], (w, h), interpolation=cv2.INTER_LINEAR)
            )
        imgs = np.stack(resized)
        assert imgs.shape == (T, h, w, C)

        assert imgs.dtype == np.uint8

        # We deliberately keep the imgs in datapoint as np.uint8 to keep the
        # cpu memory foot print small. Since dataloader multiplies by num_proc
        # x num_gpus, we want to save cpu memory. All preprocessing should be
        # done on the gpu side by the ml model.
        data[f"obs.{camera_name}"] = imgs
        
        if has_depth:
            print("Dataset has depth information")
            
            depths = observation_dict[f'{observation_key}_depth']
            depths = _repeat(depths, num=repeat_head, mode="head")
            # (T, H, W) unnormalized.
            depths = _repeat(depths, num=repeat_tail, mode="tail")

            # resize
            resized = []
            h, w = image_shapes[camera_name]
            for i in range(depths.shape[0]):
                resized.append(
                    cv2.resize(depths[i], (w, h), interpolation=cv2.INTER_LINEAR)
                )
            depths = np.stack(resized)
            assert depths.shape == (T, h, w)

            # truncate depth and normalize
            indices = np.where(depths > 2000.)
            depths[indices] = 0.
            depths = depths.astype(np.float32)
            assert depths.min() >= 0. and depths.max() <= 2000.
            
            data[f"obs.{camera_name}_depth"] = depths
        
        if has_label and f'{observation_key}_label' in observation_dict:
            print("Dataset has label information")
            
            labels = observation_dict[f'{observation_key}_label']
            labels = _repeat(labels, num=repeat_head, mode="head")
            # (T, H, W) unnormalized.
            labels = _repeat(labels, num=repeat_tail, mode="tail")

            # resize
            resized = []
            h, w = image_shapes[camera_name]
            for i in range(labels.shape[0]):
                resized.append(
                    cv2.resize(labels[i], (w, h), interpolation=cv2.INTER_NEAREST)
                )
            labels = np.stack(resized)
            assert labels.shape == (T, h, w)
            
            data[f"obs.{camera_name}_label"] = labels

    return data


def bimanual_spartan_episode_to_replay_buffer_episode(
    *,
    observation_dict,
    camera_dict,
    action_dict,
    skill_name: str,
    lowdim_keys: List[str],
    camera_names: List[str],
    camera_id_to_camera_name: Dict[str, str],
    repeat_head: int,
    repeat_tail: int,
    image_shapes: Dict[str, tuple],
    has_depth: bool,
    has_label: bool,
):
    action = action_dict["actions"].astype(np.float32)
    action = _repeat(action, num=repeat_head, mode="head")
    action = _repeat(action, num=repeat_tail, mode="tail")
    data = {
        "action": action,
        "skill_name": np.full((action.shape[0], 1), skill_name, dtype=object),
    }

    # loop through lowdim obs
    lowdim_obs = {}
    for name in lowdim_keys:
        if name not in observation_dict:
            continue
        obs = observation_dict[name].astype(np.float32)
        obs = _repeat(obs, num=repeat_head, mode="head")
        obs = _repeat(obs, num=repeat_tail, mode="tail")
        lowdim_obs[name] = obs

    for k, v in lowdim_obs.items():
        data[f"obs.{k}"] = v

    img_data = _add_spartan_episode_images_to_replay_buffer_episode(
        observation_dict=observation_dict,
        camera_names=camera_names,
        camera_id_to_camera_name=camera_id_to_camera_name,
        repeat_head=repeat_head,
        repeat_tail=repeat_tail,
        image_shapes=image_shapes,
        has_depth=has_depth,
        has_label=has_label,
    )
    data.update(img_data)
    
    for k, v in camera_dict.items():
        for camera_id in v:
            # camera_name = camera_id_to_camera_name[camera_id]
            camera_name = camera_id
            if camera_name in camera_names:
                cam_info = v[camera_id]
                if k == 'intr':
                    scale_x = image_shapes[camera_name][1] / 640
                    scale_y = image_shapes[camera_name][0] / 480
                    cam_info[0, 0] = cam_info[0, 0] * scale_x
                    cam_info[1, 1] = cam_info[1, 1] * scale_y
                    cam_info[0, 2] = cam_info[0, 2] * scale_x
                    cam_info[1, 2] = cam_info[1, 2] * scale_y
                    cam_info = _repeat([cam_info], num=action.shape[0]-1, mode="tail")
                else:
                    cam_info = _repeat(cam_info, num=repeat_head, mode="head")
                    cam_info = _repeat(cam_info, num=repeat_tail, mode="tail")
                data[f"{camera_name}.{k}"] = cam_info

    return data


def spartan_episode_to_replay_buffer_episode(
    *,
    observation_dict,
    action_dict,
    skill_name: str,
    lowdim_keys: List[str],
    camera_names: List[str],
    camera_id_to_camera_name: Dict[str, str],
    rotation_transformer,
    repeat_head: int,
    repeat_tail: int,
    has_gripper: bool,
    image_shapes: Dict[str, tuple],
):
    """
    returns a dict with:
        obs.name: o
        action: actions
    ideall we want to return a {
        obs: {
            name: o
        },
        actions: u,
    }
    but, ReplayBuffer can only take k: nparray instead of dict.
    """
    if has_gripper:
        assert "gripper_width" in lowdim_keys
    else:
        assert "gripper_width" not in lowdim_keys

    # actions are ActionEndEffectorScripted, so we take the first 6 cols
    action = _change_pose_repr(
        action_dict["actions"][:, :6], rotation_transformer
    )
    if has_gripper:
        action = np.concatenate(
            [action, action_dict["actions"][:, 7:8].astype(np.float32)],
            axis=-1,
        )
    action = _repeat(action, num=repeat_head, mode="head")
    action = _repeat(action, num=repeat_tail, mode="tail")
    data = {
        "action": action,
        "skill_name": np.full((action.shape[0], 1), skill_name, dtype=object),
    }

    # loop through lowdim obs
    lowdim_obs = {}
    for name in lowdim_keys:
        if name == "gripper_width":
            obs = observation_dict[name].astype(np.float32)
        else:
            # assuming pose
            obs = _change_pose_repr(
                observation_dict[name], rotation_transformer
            )
        obs = _repeat(obs, num=repeat_head, mode="head")
        obs = _repeat(obs, num=repeat_tail, mode="tail")
        lowdim_obs[name] = obs

    for k, v in lowdim_obs.items():
        data[f"obs.{k}"] = v
    img_data = _add_spartan_episode_images_to_replay_buffer_episode(
        observation_dict=observation_dict,
        camera_names=camera_names,
        camera_id_to_camera_name=camera_id_to_camera_name,
        repeat_head=repeat_head,
        repeat_tail=repeat_tail,
        image_shapes=image_shapes,
    )
    data.update(img_data)

    return data


def _unflatten_dict_for_obs(dict_data: Dict) -> Dict:
    """
    Change the key-value structures only for the keys start with "obs." as
    follows:

       dict_data["obs.some_name"] -> dict_data["obs"]["some_name"]
    """
    obs = {}
    to_delete = []
    for k, v in dict_data.items():
        if k.startswith("obs."):
            new_key = k[4:]
            assert new_key not in obs
            obs[new_key] = v
            to_delete.append(k)
    for k in to_delete:
        del dict_data[k]
    dict_data["obs"] = obs
    return dict_data


class SpartanPointmapReconDataset(SpartanBaseDataset):
    def __init__(
        self,
        episode_path_globs: str,
        shape_meta: dict,
        imagenet_normalization: bool,
        horizon=1,
        n_obs_steps=1,
        pad_before=0,
        pad_after=0,
        repeat_head=0,
        repeat_tail=0,
        stride=1,
        has_gripper=False,
        rotation_rep="rotation_6d",
        val_ratio=0.0,
        seed=42,
        mode="np",
        compressor="blosc",
        max_num_episodes: Optional[int] = None,
        raw_rgb=True,
        num_workers=8,
        replay_buffer_path=None,
        is_multiarm=False,
        is_relative=False,
        path_is_fully_resolved=False,
        apply_static_filter=False,
        vae_3d_encoder=False,
        has_depth=False,
        has_label=False,
    ):
        super().__init__(
            episode_path_globs=episode_path_globs,
            shape_meta=shape_meta,
            imagenet_normalization=imagenet_normalization,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            repeat_head=repeat_head,
            repeat_tail=repeat_tail,
            has_gripper=has_gripper,
            rotation_rep=rotation_rep,
            val_ratio=val_ratio,
            stride=stride,
            seed=seed,
            mode=mode,
            compressor=compressor,
            max_num_episodes=max_num_episodes,
            num_workers=num_workers,
            replay_buffer_path=replay_buffer_path,
            is_multiarm=is_multiarm,
            is_relative=is_relative,
            path_is_fully_resolved=path_is_fully_resolved,
            raw_rgb=raw_rgb,
            has_depth=has_depth,
            has_label=has_label
        )
        self.stride = stride
        self.apply_static_filter = apply_static_filter
        self.vae_3d_encoder = vae_3d_encoder

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx, self.apply_static_filter, self.stride)

        if self._is_relative:
            data = change_to_relative_trajectories(
                data=data,
                # This is the time index of "T=0" or "now". The indices
                # preceding this value are in the past and those after this
                # value are in the future.
                base_index=self.n_obs_steps - 1,
                shape_meta=self._shape_meta,
            )

        # Pop skill_name from data before converting to torch because
        # skill_name is a string object and can't be converted to a tensor.
        if "skill_name" in data:
            skill_name_array = data.pop("skill_name")

        torch_data = dict_apply(data, torch.from_numpy)

        if f"obs.{self._camera_names[0]}" in torch_data:
            if not self._raw_rgb:
                for camera in self._camera_names:
                    torch_data[f"obs.{camera}"] = (
                        torch_data[f"obs.{camera}"]
                        .permute(0, 3, 1, 2)
                        .contiguous()
                        .float()
                        / 255.0
                        * 2.0 - 1.0
                    )

            mv_torch_data = dict()
            selected_camera_names = np.random.choice(self._camera_names, 2, replace=False)
            for idx, camera in enumerate(selected_camera_names):
                key = f"obs.{camera}"
                images = torch_data[key]
                
                depth_key = f"obs.{camera}_depth"
                depth_images = torch_data[depth_key]
                
                cam_intr = torch_data[f"{camera}.intr"]
                cam_extr = torch_data[f"{camera}.extr"]

                # NOTE: In vanilla SVD task, first frame is given, all predicted are next.
                pointmaps = depth_imgs_to_maps(depth_images, cam_intr[0])
                pointmaps = torch.tensor(pointmaps).permute(0, 3, 1, 2)
                pointmaps_min, pointmaps_max = -1, 2
                normalized_pointmaps = ((pointmaps - pointmaps_min) / (pointmaps_max - pointmaps_min)) * 2 - 1

                if idx == 0:
                    if self.vae_3d_encoder:
                        normalized_pointmaps = rearrange(normalized_pointmaps, 't c h w -> c t h w')
                        normalized_pointmaps = normalized_pointmaps
                    else:
                        normalized_pointmaps = normalized_pointmaps
                    
                    sv_torch_data = {
                        "image": images.float(),
                        "pointmap": normalized_pointmaps.float(),
                        "cam_intr": cam_intr,
                        "cam_extr": cam_extr,
                    }
                else:
                    # change view 2 to view 1's camera coordinate system
                    camera_transform = torch.linalg.inv(sv_torch_data["cam_extr"]) @ cam_extr
                    converted_pointmaps = apply_transform_on_depthmaps(pointmaps.permute(0, 2, 3, 1), camera_transform)
                    converted_pointmaps = converted_pointmaps.permute(0, 3, 1, 2)
                    
                    pointmaps_min, pointmaps_max = -1, 2
                    normalized_converted_pointmaps = ((converted_pointmaps - pointmaps_min) / (pointmaps_max - pointmaps_min)) * 2 - 1
                    
                    # mask = ~torch.all(depth_maps[:1]==0., axis=1)

                    if self.vae_3d_encoder:
                        normalized_converted_pointmaps = rearrange(normalized_converted_pointmaps, 't c h w -> c t h w')
                        normalized_converted_pointmaps = normalized_converted_pointmaps
                    else:
                        normalized_converted_pointmaps = normalized_converted_pointmaps
                    
                    sv_torch_data = {
                        "image_right": images.float(),
                        "pointmap_right": normalized_converted_pointmaps.float(),
                    }
                mv_torch_data.update(sv_torch_data)
            
            n_frames = images.shape[0]
            motion_bucket_id = torch.ones(size=(n_frames,)) * 127
            fps_id = torch.ones(size=(n_frames,)) * 6
            image_only_indicator = torch.zeros(size=(1, n_frames,))
            addn_data = {
                "motion_bucket_id": motion_bucket_id,
                "fps_id": fps_id,
                "image_only_indicator": image_only_indicator,
                "idx": idx,
                "num_video_frames": n_frames,
            }
            mv_torch_data.update(addn_data)
            
            return mv_torch_data

        return _unflatten_dict_for_obs(torch_data)

    def _make_relative_proprioception_normalier_params(self):
        # this impl iterates through the entire dataset to
        # compute the min and max for each dimension, which is pretty
        # unnecessary and won't scale to webdatasets. Should run this on
        # some large subset of our data, and have a fixed normalizer instead.
        normalizer = LinearNormalizer()
        keys = [f"obs.{k}" for k in self._lowdim_names] + ["action"]
        data_cache = {key: list() for key in keys}
        self.sampler.limit_keys(keys)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=8,
        )
        for batch in tqdm(
            dataloader, desc="iterating dataset to get normalization"
        ):
            for key in self._lowdim_names:
                data_cache[f"obs.{key}"].append(
                    copy.deepcopy(batch["obs"][key].numpy())
                )
            data_cache["action"].append(copy.deepcopy(batch["action"].numpy()))
        self.sampler.limit_keys(None)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            data_cache[key] = data_cache[key].reshape(B * T, D)

        # action
        action_normalizers = list()
        action_start_idx = {"left": 9, "right": 0}
        for arm in ["left", "right"]:
            start_idx = action_start_idx[arm]
            # pos
            action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(
                        data_cache["action"][:, start_idx : start_idx + 3]
                    )
                )
            )
            # rot
            action_normalizers.append(
                get_identity_normalizer_from_stat(
                    array_to_stats(
                        data_cache["action"][:, start_idx + 3 : start_idx + 9]
                    )
                )
            )
        # gripper
        action_normalizers.append(
            get_range_normalizer_from_stat(
                array_to_stats(data_cache["action"][:, 18:])
            )
        )
        normalizer["action"] = concatenate_normalizer(action_normalizers)

        # obs
        for key in self._lowdim_names:
            stat = array_to_stats(data_cache[f"obs.{key}"])

            if key.endswith("xyz"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("rot_6d"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("panda_hand"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer
        return normalizer

    def _make_absolute_proprioception_normalier_params(self):
        normalizer = LinearNormalizer()

        # normalizer for lowdim and actions
        stuff_to_normalize = self._lowdim_names + ["action"]
        for name in stuff_to_normalize:
            replay_buffer_name = name
            if name in self._lowdim_names:
                replay_buffer_name = f"obs.{name}"

            scale, offset, info = get_normalizer_params(
                self.replay_buffer[replay_buffer_name]
            )
            normalizer[name] = SingleFieldLinearNormalizer.create_manual(
                scale=scale,
                offset=offset,
                input_stats_dict=info,
            )

        return normalizer

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        if self._is_relative:
            normalizer = self._make_relative_proprioception_normalier_params()
        else:
            normalizer = self._make_absolute_proprioception_normalier_params()

        # image
        for camera_name in self._camera_names:
            normalizer[camera_name] = (
                SingleFieldLinearNormalizer.create_identity()
            )

        # if i am a feature based dataset
        assert (
            "obs.feature" not in self.replay_buffer
        ), "Shouldn't call this for feature dataset"

        return normalizer


class SpartanVideoMultiViewDataset(SpartanBaseDataset):
    def __init__(
        self,
        episode_path_globs: str,
        shape_meta: dict,
        imagenet_normalization: bool,
        horizon=1,
        n_obs_steps=1,
        pad_before=0,
        pad_after=0,
        repeat_head=0,
        repeat_tail=0,
        stride=1,
        has_gripper=False,
        rotation_rep="rotation_6d",
        val_ratio=0.0,
        seed=42,
        mode="np",
        compressor="blosc",
        max_num_episodes: Optional[int] = None,
        raw_rgb=True,
        num_workers=8,
        replay_buffer_path=None,
        is_multiarm=False,
        is_relative=False,
        path_is_fully_resolved=False,
        apply_static_filter=False,
        has_depth=False,
        has_label=False,
    ):
        super().__init__(
            episode_path_globs=episode_path_globs,
            shape_meta=shape_meta,
            imagenet_normalization=imagenet_normalization,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            repeat_head=repeat_head,
            repeat_tail=repeat_tail,
            has_gripper=has_gripper,
            rotation_rep=rotation_rep,
            val_ratio=val_ratio,
            stride=stride,
            seed=seed,
            mode=mode,
            compressor=compressor,
            max_num_episodes=max_num_episodes,
            num_workers=num_workers,
            replay_buffer_path=replay_buffer_path,
            is_multiarm=is_multiarm,
            is_relative=is_relative,
            path_is_fully_resolved=path_is_fully_resolved,
            raw_rgb=raw_rgb,
            has_depth=has_depth,
            has_label=has_label
        )
        self.stride = stride
        self.apply_static_filter = apply_static_filter

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx, self.apply_static_filter, self.stride)

        if self._is_relative:
            data = change_to_relative_trajectories(
                data=data,
                # This is the time index of "T=0" or "now". The indices
                # preceding this value are in the past and those after this
                # value are in the future.
                base_index=self.n_obs_steps - 1,
                shape_meta=self._shape_meta,
            )

        # Pop skill_name from data before converting to torch because
        # skill_name is a string object and can't be converted to a tensor.
        if "skill_name" in data:
            skill_name_array = data.pop("skill_name")

        torch_data = dict_apply(data, torch.from_numpy)

        if f"obs.{self._camera_names[0]}" in torch_data:
            if not self._raw_rgb:
                for camera in self._camera_names:
                    torch_data[f"obs.{camera}"] = (
                        torch_data[f"obs.{camera}"]
                        .permute(0, 3, 1, 2)
                        .contiguous()
                        .float()
                        / 255.0
                        * 2.0 - 1.0
                    )

            mv_torch_data = dict()
            selected_camera_names = np.random.choice(self._camera_names, 2, replace=False)
            n_cond_frames = 1
            for idx, camera in enumerate(selected_camera_names):
                key = f"obs.{camera}"
                images = torch_data[key]
                
                depth_key = f"obs.{camera}_depth"
                depth_images = torch_data[depth_key]
                
                label_key = f'obs.{camera}_label'
                labels = torch_data[label_key]
                
                masks = torch.isin(labels, torch.tensor([29, 30, 31, 33, 34, 35]))
                
                dilated_masks = []
                for mask in masks:
                    resized_mask = cv2.resize(mask.cpu().numpy().astype(np.uint8), (mask.shape[1]//8, mask.shape[0]//8), interpolation=cv2.INTER_NEAREST)
                    kernel = np.ones((2, 2), np.uint8)
                    dilated_mask = cv2.dilate(resized_mask, kernel, iterations=2)
                    dilated_masks.append(dilated_mask)
                    
                dilated_masks = torch.tensor(np.array(dilated_masks)).unsqueeze(1)
                
                cam_intr = torch_data[f"{camera}.intr"]
                cam_extr = torch_data[f"{camera}.extr"]

                # NOTE: In vanilla SVD task, first frame is given, all predicted are next.
                pointmaps = depth_imgs_to_maps(depth_images, cam_intr[0])
                pointmaps = torch.tensor(pointmaps).permute(0, 3, 1, 2)
                pointmaps_min, pointmaps_max = -1, 2
                normalized_pointmaps = ((pointmaps - pointmaps_min) / (pointmaps_max - pointmaps_min)) * 2 - 1
                
                pointmaps_with_color = torch.cat([normalized_pointmaps, images], dim=1)

                if idx == 0:
                    n_frames = pointmaps_with_color.shape[0] - n_cond_frames
                    cond_aug = torch.ones(size=(n_cond_frames,)) * 0.02
                    cond_pointmaps_without_noise = normalized_pointmaps[:n_cond_frames].repeat(n_frames, 1, 1, 1)
                    cond_images_without_noise = images[:n_cond_frames].repeat(n_frames, 1, 1, 1)
                    cond_pointmaps = (
                        cond_pointmaps_without_noise
                        + 0.02 * torch.randn(*cond_pointmaps_without_noise.shape))
                    cond_images = (
                        cond_images_without_noise
                        + 0.02 * torch.randn(*cond_images_without_noise.shape))
                    
                    pointmaps_with_color = pointmaps_with_color[n_cond_frames:]
                    dilated_masks = dilated_masks[n_cond_frames:]
                    
                    sv_torch_data = {
                        "pointmap": pointmaps_with_color.float(),
                        "masks": dilated_masks,
                        # "cond_pointmaps": cond_pointmaps.float(),
                        "cond_pointmaps_without_noise": cond_pointmaps_without_noise.float(),
                        # "cond_images": cond_images.float(),
                        "cond_images_without_noise": cond_images_without_noise.float(),
                        "cam_intr": cam_intr,
                        "cam_extr": cam_extr,
                    }
                else:
                    # change view 2 to view 1's camera coordinate system
                    camera_transform = torch.linalg.inv(sv_torch_data["cam_extr"]) @ cam_extr
                    converted_pointmaps = apply_transform_on_depthmaps(pointmaps.permute(0, 2, 3, 1), camera_transform)
                    converted_pointmaps = converted_pointmaps.permute(0, 3, 1, 2)
                    
                    pointmaps_min, pointmaps_max = -1, 2
                    normalized_converted_pointmaps = ((converted_pointmaps - pointmaps_min) / (pointmaps_max - pointmaps_min)) * 2 - 1
                    converted_pointmaps_with_color = torch.cat([normalized_converted_pointmaps, images], dim=1)
                    
                    # mask = ~torch.all(depth_maps[:1]==0., axis=1)
                    
                    n_frames = converted_pointmaps_with_color.shape[0] - n_cond_frames
                    cond_aug = torch.ones(size=(n_cond_frames,)) * 0.02
                    cond_pointmaps_without_noise = normalized_pointmaps[:1].repeat(n_frames, 1, 1, 1)
                    cond_images_without_noise = images[:1].repeat(n_frames, 1, 1, 1)
                    cond_pointmaps = (
                        cond_pointmaps_without_noise
                        + 0.02 * torch.randn(*cond_pointmaps_without_noise.shape))
                    cond_images = (
                        cond_images_without_noise
                        + 0.02 * torch.randn(*cond_images_without_noise.shape))
                    
                    converted_pointmaps_with_color = converted_pointmaps_with_color[n_cond_frames:]
                    dilated_masks = dilated_masks[n_cond_frames:]
                    
                    sv_torch_data = {
                        "pointmap_right": converted_pointmaps_with_color.float(),
                        "masks_right": dilated_masks,
                        "cam_extr_right": cam_extr,
                        # "cond_pointmaps_right": cond_pointmaps.float(),
                        "cond_pointmaps_without_noise_right": cond_pointmaps_without_noise.float(),
                        # "cond_images_right": cond_images.float(),
                        "cond_images_without_noise_right": cond_images_without_noise.float(),
                    }
                mv_torch_data.update(sv_torch_data)
            
            n_frames = mv_torch_data["pointmap"].shape[0]
            motion_bucket_id = torch.ones(size=(n_frames,)) * 127
            fps_id = torch.ones(size=(n_frames,)) * 6
            image_only_indicator = torch.zeros(size=(1, n_frames,))
            addn_data = {
                "motion_bucket_id": motion_bucket_id,
                "fps_id": fps_id,
                "image_only_indicator": image_only_indicator,
                "idx": idx,
                "num_video_frames": n_frames,
            }
            mv_torch_data.update(addn_data)
            
            return mv_torch_data

        return _unflatten_dict_for_obs(torch_data)


    def _make_relative_proprioception_normalier_params(self):
        normalizer = LinearNormalizer()
        keys = [f"obs.{k}" for k in self._lowdim_names] + ["action"]
        data_cache = {key: list() for key in keys}
        self.sampler.limit_keys(keys)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=8,
        )
        for batch in tqdm(
            dataloader, desc="iterating dataset to get normalization"
        ):
            for key in self._lowdim_names:
                data_cache[f"obs.{key}"].append(
                    copy.deepcopy(batch["obs"][key].numpy())
                )
            data_cache["action"].append(copy.deepcopy(batch["action"].numpy()))
        self.sampler.limit_keys(None)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            data_cache[key] = data_cache[key].reshape(B * T, D)

        # action
        action_normalizers = list()
        action_start_idx = {"left": 9, "right": 0}
        for arm in ["left", "right"]:
            start_idx = action_start_idx[arm]
            # pos
            action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(
                        data_cache["action"][:, start_idx : start_idx + 3]
                    )
                )
            )
            # rot
            action_normalizers.append(
                get_identity_normalizer_from_stat(
                    array_to_stats(
                        data_cache["action"][:, start_idx + 3 : start_idx + 9]
                    )
                )
            )
        # gripper
        action_normalizers.append(
            get_range_normalizer_from_stat(
                array_to_stats(data_cache["action"][:, 18:])
            )
        )
        normalizer["action"] = concatenate_normalizer(action_normalizers)

        # obs
        for key in self._lowdim_names:
            stat = array_to_stats(data_cache[f"obs.{key}"])

            if key.endswith("xyz"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("rot_6d"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("panda_hand"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer
        return normalizer

    def _make_absolute_proprioception_normalier_params(self):
        normalizer = LinearNormalizer()

        # normalizer for lowdim and actions
        stuff_to_normalize = self._lowdim_names + ["action"]
        for name in stuff_to_normalize:
            replay_buffer_name = name
            if name in self._lowdim_names:
                replay_buffer_name = f"obs.{name}"

            scale, offset, info = get_normalizer_params(
                self.replay_buffer[replay_buffer_name]
            )
            normalizer[name] = SingleFieldLinearNormalizer.create_manual(
                scale=scale,
                offset=offset,
                input_stats_dict=info,
            )

        return normalizer

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        if self._is_relative:
            normalizer = self._make_relative_proprioception_normalier_params()
        else:
            normalizer = self._make_absolute_proprioception_normalier_params()

        # image
        for camera_name in self._camera_names:
            normalizer[camera_name] = (
                SingleFieldLinearNormalizer.create_identity()
            )

        # if i am a feature based dataset
        assert (
            "obs.feature" not in self.replay_buffer
        ), "Shouldn't call this for feature dataset"

        return normalizer