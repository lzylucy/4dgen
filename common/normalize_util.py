import pickle
from typing import Dict, List

import numpy as np
import torch

from common.fsspec_util import load_file_with_fsspec
from common.pytorch_util import (
    dict_apply,
    dict_apply_reduce,
    dict_apply_split,
)
from model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


def merge_stats_dicts(
    stats_dict_a: Dict[str, Dict[str, np.ndarray]],
    stats_dict_b: Dict[str, Dict[str, np.ndarray]],
    skip_episode_paths: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Merge two stats dicts together. For merging standard deviations, the pooled
    variance method is used: https://en.wikipedia.org/wiki/Pooled_variance.

    Args:
        stats_dict_a, stats_dict_b:
            Dict with the following key-value pairs:
                stats:
                    Dict where the keys are lowdim names such as `action`,
                    `robot__desired__poses_right::panda__xyz`, etc., and
                    the values are the corresponding stats dicts, i.e.
                    {min: np.ndarray, max: np.ndarray, mean: np.ndarray,
                    std: np.ndarray}.
                episode_paths:
                    List of episode paths used to compute the stats.
                num_datapoints:
                    Number of data used to compute the stats.
        skip_episode_paths:
            If True, it skips merging `episode_paths` and the resulting
            `episode_paths` entry will be an empty list. The default is False.
    Return:
        New merged stats in the same data format as the inputs to the function.
    """
    episode_paths = []
    stats_dicts = (stats_dict_a, stats_dict_b)
    if not skip_episode_paths:
        for d in stats_dicts:
            episode_paths += d["episode_paths"]

    stats = [d["stats"] for d in stats_dicts]
    sample_sizes = [d["num_datapoints"] for d in stats_dicts]
    merged = merge_stats(stats, sample_sizes)
    return dict(
        stats=merged,
        num_datapoints=sum(sample_sizes),
        episode_paths=episode_paths,
    )


def merge_stats(
    list_of_stats: List[Dict[str, Dict[str, np.ndarray]]],
    list_of_sample_sizes: List[int],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Merge two stats together. For merging standard deviations, the pooled
    variance method is used: https://en.wikipedia.org/wiki/Pooled_variance.
    Args:
        list_of_stats:
            List of normalizer stats where each list item contains a dict whose
            keys are lowdim names such as `action`,
            `robot__desired__poses_right::panda__xyz`, etc., and whose values
            are the corresponding stats dicts, i.e. {min: np.ndarray,
            max: np.ndarray, mean: np.ndarray, std: np.ndarray}.
        list_of_sample_sizes:
            List of sample size where the i-th list item denotes the sample
            size used to compute the i-th normalizer stats in `list_of_stats`.
    Return:
        New merged stats in the same data format as that of the items in
        `list_of_stats`.
    """
    keys = set(list_of_stats[0].keys())
    for stats in list_of_stats:
        assert keys == set(stats.keys())

    # (B, 1)
    sample_sizes = np.array(list_of_sample_sizes)[:, None]
    new_stats = {}
    for k in keys:
        new_min = np.stack([stats[k]["min"] for stats in list_of_stats]).min(
            axis=0
        )
        new_max = np.stack([stats[k]["max"] for stats in list_of_stats]).max(
            axis=0
        )

        means = np.stack([stats[k]["mean"] for stats in list_of_stats])
        new_mean = (means * sample_sizes).sum(axis=0) / sample_sizes.sum()
        new_mean = new_mean.astype(means.dtype)

        # we use this: https://en.wikipedia.org/wiki/Pooled_variance
        stds = np.stack([stats[k]["std"] for stats in list_of_stats])
        variances = np.power(stds, 2)
        new_variances = (variances * (sample_sizes - 1)).sum(axis=0) / (
            sample_sizes.sum() - sample_sizes.shape[0]
        )
        new_std = np.sqrt(new_variances)
        new_std = new_std.astype(stds.dtype)
        new_stats[k] = {
            "min": new_min,
            "max": new_max,
            "mean": new_mean,
            "std": new_std,
        }

    return new_stats


def get_linear_normalizer_from_saved_stats(stats_path):
    """
    Returns a LinearNormalizer built from stats stored in the pkl file at
    `stats_path`.
    """
    print(f"Loading normalizer stats from: {stats_path}")
    # This handles both local and S3 files.
    data = load_file_with_fsspec(stats_path, pickle.load, mode="rb")
    stats = data["stats"]

    normalizer = LinearNormalizer()
    for k, v in stats.items():
        normalizer[k] = get_range_normalizer_from_stat(v)
    return normalizer


def save_stats_to_pickle(
    *,
    pkl_path: str,
    stats: Dict[str, Dict[str, np.ndarray]],
    num_datapoints: int,
    episode_paths: List[str],
):
    """
    Saves a dict of `stats`, `num_datapoints` and `episode_paths` as a pickle
    file to `pkl_path`.
    Args:
        pkl_path: abs path to save the pickle file
        stats: dict from lowdim key name to a stats dict that has the following
            elements of the corresponding dimensions: min, max, mean, std
        num_datapoints: number of data points used to compute stats.
        episode_paths: List of spartan paths used to compute stats.
    """
    data = {
        "stats": stats,
        "episode_paths": episode_paths,
        "num_datapoints": num_datapoints,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)


def get_range_normalizer_from_stat(
    stat, output_max=1, output_min=-1, range_eps=1e-7
):
    # -1, 1 normalization
    input_max = stat["max"]
    input_min = stat["min"]
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_image_range_normalizer():
    scale = np.array([2], dtype=np.float32)
    offset = np.array([-1], dtype=np.float32)
    stat = {
        "min": np.array([0], dtype=np.float32),
        "max": np.array([1], dtype=np.float32),
        "mean": np.array([0.5], dtype=np.float32),
        "std": np.array([np.sqrt(1 / 12)], dtype=np.float32),
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_identity_normalizer_from_stat(stat):
    scale = np.ones_like(stat["min"])
    offset = np.zeros_like(stat["min"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def array_to_stats(arr: np.ndarray):
    stat = {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
    }
    return stat


def concatenate_normalizer(normalizers: list):
    scale = torch.concatenate(
        [normalizer.params_dict["scale"] for normalizer in normalizers],
        axis=-1,
    )
    offset = torch.concatenate(
        [normalizer.params_dict["offset"] for normalizer in normalizers],
        axis=-1,
    )
    input_stats_dict = dict_apply_reduce(
        [normalizer.params_dict["input_stats"] for normalizer in normalizers],
        lambda x: torch.concatenate(x, axis=-1),
    )
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=input_stats_dict
    )
