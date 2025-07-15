from typing import Optional

import numba
import numpy as np

from common.replay_buffer import ReplayBuffer


LEFT_ARM_POS_KEY = 'obs.robot__actual__poses__left::panda__xyz'
RIGHT_ARM_POS_KEY = 'obs.robot__actual__poses__right::panda__xyz'
ROBOT_FREQUENCY = 10


@numba.jit(nopython=True)
def create_indices(
    episode_end_indices: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
    debug: bool = True,
) -> np.ndarray:
    """
    Create an numpy.ndarray of indices where each array item consists of the
    following four indices:
        (buffer_start, buffer_end, sample_start, sample_end)

        buffer_start:
            The start index for this sample in the entire episode trajectory.
        buffer_end:
            The end index for this sample in the entire episode trajectory.
        sample_start:
            The start index pointing where the actual data starts in the
            sample. The sample data before this index will be filled with the
            first actual data in the sample.
        sample_end:
            The end index pointing where the actual data ends in the sample.
            The sample data after this index will be filled with the last
            actual data in the sample.

    Example:
        Inputs:
            episode_end_indices = [76, any, any, ...]
            sequence_length = 16
            episode_mask = [1, 0, 0, ...]
            pad_before = 1
            pad_after = 7

        Output:
            [
                [ 0 15  1 16],  # <- pad_before
                [ 0 16  0 16],
                [ 1 17  0 16],
                [ 2 18  0 16],
                [ 3 19  0 16],
                ...
                [58 74  0 16],
                [59 75  0 16],
                [60 76  0 16],
                [61 76  0 15],  # <- pad_after
                [62 76  0 14],  # <- pad_after
                [63 76  0 13],  # <- pad_after
                [64 76  0 12],  # <- pad_after
                [65 76  0 11],  # <- pad_after
                [66 76  0 10],  # <- pad_after
                [67 76  0  9],  # <- pad_after
            ]

    Args:
        episode_end_indices:
            Array of indices representing the ends of episodes for the
            multi-episode data.
        sequence_length:
            Length of a sequence. This is also referred to as horizon. This
            must be >= 1.
        episode_mask:
            Array of flags representing whether the episode in question will
            be included or not. True (or 1) means including it.
        pad_before:
            Number of padding before the beginning of the actual data.
        pad_after:
            Number of padding after the end of the actual data.
        debug:
            If True, it goes through several asserts.
    """
    assert sequence_length >= 1
    assert episode_mask.shape == episode_end_indices.shape
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for index in range(len(episode_end_indices)):
        if not episode_mask[index]:
            # skip episode
            continue
        episode_start_idx = 0
        if index > 0:
            episode_start_idx = episode_end_indices[index - 1]
        episode_end_idx = episode_end_indices[index]
        episode_length = episode_end_idx - episode_start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + episode_start_idx
            buffer_end_idx = (
                min(idx + sequence_length, episode_length) + episode_start_idx
            )
            start_offset = buffer_start_idx - (idx + episode_start_idx)
            end_offset = (
                idx + sequence_length + episode_start_idx
            ) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert start_offset >= 0
                assert end_offset >= 0
                assert (sample_end_idx - sample_start_idx) == (
                    buffer_end_idx - buffer_start_idx
                )
            indices.append(
                [
                    buffer_start_idx,
                    buffer_end_idx,
                    sample_start_idx,
                    sample_end_idx,
                ]
            )
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


class SequenceSampler:
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray] = None,
    ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert sequence_length >= 1
        if keys is None:
            keys = list(replay_buffer.keys())

        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(
                episode_ends,
                sequence_length=sequence_length,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=episode_mask,
            )
        else:
            indices = np.zeros((0, 4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices
        self.keys = list(keys)  # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k

        self.limited_keys = (
            None  # speed up the interation when getting normalizaer
        )

    def __len__(self):
        return len(self.indices)

    def limit_keys(self, keys=None):
        """
        After calling this, the returned sample will only contain keys that are
        subset of `keys`. Set `keys` to None to return all data.
        """
        if keys is not None:
            self.limited_keys = [k for k in keys if k in self.keys]
        else:
            self.limited_keys = keys

    def sample_sequence(self, idx, apply_static_filter, stride):
        while True:
            buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
                self.indices[idx]
            )
            result = dict()

            if self.limited_keys is None:
                keys = self.keys
            else:
                keys = self.limited_keys

            for key in keys:
                input_arr = self.replay_buffer[key]
                # performance optimization, avoid small allocation if possible
                if key not in self.key_first_k:
                    sample = input_arr[buffer_start_idx:buffer_end_idx]
                else:
                    # performance optimization, only load used obs steps
                    n_data = buffer_end_idx - buffer_start_idx
                    k_data = min(self.key_first_k[key], n_data)
                    # fill value with Nan to catch bugs
                    # the non-loaded region should never be used
                    sample = np.full(
                        (n_data,) + input_arr.shape[1:],
                        fill_value=np.nan,
                        dtype=input_arr.dtype,
                    )
                    try:
                        sample[:k_data] = input_arr[
                            buffer_start_idx : buffer_start_idx + k_data
                        ]
                    except Exception as e:
                        import pdb

                        pdb.set_trace()
                data = sample
                if (sample_start_idx > 0) or (
                    sample_end_idx < self.sequence_length
                ):
                    data = np.zeros(
                        shape=(self.sequence_length,) + input_arr.shape[1:],
                        dtype=input_arr.dtype,
                    )
                    if sample_start_idx > 0:
                        data[:sample_start_idx] = sample[0]
                    if sample_end_idx < self.sequence_length:
                        data[sample_end_idx:] = sample[-1]
                    data[sample_start_idx:sample_end_idx] = sample
                result[key] = data
                
            if apply_static_filter:
                left_arm_pos = result[LEFT_ARM_POS_KEY]
                right_arm_pos = result[RIGHT_ARM_POS_KEY]
                left_vel = (np.linalg.norm(left_arm_pos[4] - left_arm_pos[0], 2))/(4/(ROBOT_FREQUENCY // stride))
                right_vel = (np.linalg.norm(right_arm_pos[4] - right_arm_pos[0], 2))/(4/(ROBOT_FREQUENCY // stride))
                
                if left_vel > 0.02 or right_vel > 0.02:
                    break
                else:
                    idx = np.random.choice(len(self.indices))
            else:
                break
        return result
