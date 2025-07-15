import numpy as np

from common.pose_util import (
    convert_pose_mat_rep,
    mat_to_pose9d,
    pose9d_to_mat,
)


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    else:
        return x.cpu().detach().numpy()


def _get_pose_mat(data, arm, obs_prefix="obs."):
    arm_string = f"{obs_prefix}robot__actual__poses__{arm}"

    return pose9d_to_mat(
        np.concatenate(
            [
                _to_numpy(data[f"{arm_string}::panda__xyz"]),
                _to_numpy(data[f"{arm_string}::panda__rot_6d"]),
            ],
            axis=-1,
        )
    )


def change_to_relative_trajectories(
    *,
    data,
    base_index,
    shape_meta,
    obs_prefix="obs.",
):
    """
    Converts the absolute trajectory representation in `data` to a relative
    trajectory representation wrt to the `base_index` timestep's
    transformation. This converts both the states (observations) and the
    actions. This also adds a relative pose between the two arms. This assumes
    spartan bimanual dataformat in `data`.

    """

    X_W_arm_traj = {}
    result = {}
    for arm in ["left", "right"]:
        X_W_arm_traj[arm] = _get_pose_mat(data, arm, obs_prefix)

    # add relative between left and right
    for arm in ["left", "right"]:
        other_arm = "right" if arm == "left" else "left"
        X_W_other_traj = _get_pose_mat(data, other_arm, obs_prefix)

        X_other_arm_traj = convert_pose_mat_rep(
            X_W_arm_traj[arm],
            X_A_C=X_W_other_traj[base_index],
            pose_rep="relative",
            backward=False,
        )
        vec_other_arm_traj = mat_to_pose9d(X_other_arm_traj)
        # X_other_me
        result[
            f"{obs_prefix}robot__actual__poses__{other_arm}__{arm}::panda__xyz"
        ] = vec_other_arm_traj[:, :3]
        result[
            f"{obs_prefix}robot__actual__poses__{other_arm}__{arm}::panda__rot_6d"  # noqa
        ] = vec_other_arm_traj[:, 3:]

    # convert pose and action to relative
    # actions are [right_9d_pose, left_9d_pose, right_gripper, left_gripper]
    action_start_idx = {"left": 9, "right": 0}
    if "action" in data:
        result["action"] = _to_numpy(data["action"]).copy()

    for arm in ["left", "right"]:
        # change base frame to current time steps' arm pose
        X_cur_arm_traj = convert_pose_mat_rep(
            X_W_arm_traj[arm],
            X_A_C=X_W_arm_traj[arm][base_index],
            pose_rep="relative",
            backward=False,
        )
        vec_cur_arm_traj = mat_to_pose9d(X_cur_arm_traj)
        # overwrite obs
        result[f"{obs_prefix}robot__actual__poses__{arm}::panda__xyz"] = (
            vec_cur_arm_traj[:, :3]
        )
        result[f"{obs_prefix}robot__actual__poses__{arm}::panda__rot_6d"] = (
            vec_cur_arm_traj[:, 3:]
        )

        if "action" in data:
            start_idx = action_start_idx[arm]
            X_W_action_traj = pose9d_to_mat(
                _to_numpy(data["action"][:, start_idx : start_idx + 9])
            )
            # change base frame to current time steps' arm pose
            X_cur_action_traj = convert_pose_mat_rep(
                X_W_action_traj,
                X_A_C=X_W_arm_traj[arm][base_index],
                pose_rep="relative",
                backward=False,
            )

            # overwrite action
            vec_cur_action_traj = mat_to_pose9d(X_cur_action_traj)
            result["action"][:, start_idx : start_idx + 3] = (
                vec_cur_action_traj[:, :3]
            )
            result["action"][:, start_idx + 3 : start_idx + 9] = (
                vec_cur_action_traj[:, 3:]
            )

    # add shallow copy of everything else that's in data but not in result
    # into result
    missing_keys = [k for k in data.keys() if k not in result]
    for k in missing_keys:
        result[k] = data[k]

    return result


def change_rel_action_to_abs(
    *,
    X_W_arm,
    vec_cur_action_traj,
):
    """
    Converts relative action trajectory back to world frame using current end
    effector poses in the world frame.
    Args:
        X_W_arm: dict from arm name to 4d transformation matrix. Keys should
            be [left::panda, right::panda]
        vec_cur_action_traj: (N, 20) actions. The layout need to match spartan
            format
    """

    # go from relative tback to absolute
    action_start_idx = {"left::panda": 9, "right::panda": 0}
    vec_W_action_traj = vec_cur_action_traj.copy()
    for arm in ["left::panda", "right::panda"]:
        start_idx = action_start_idx[arm]
        X_cur_action_traj = pose9d_to_mat(
            vec_cur_action_traj[:, start_idx : start_idx + 9]
        )
        X_W_action_traj = convert_pose_mat_rep(
            X_cur_action_traj,
            X_A_C=X_W_arm[arm],
            pose_rep="relative",
            backward=True,
        )
        my_vec_W_action_traj = mat_to_pose9d(X_W_action_traj)
        vec_W_action_traj[:, start_idx : start_idx + 3] = my_vec_W_action_traj[
            :, :3
        ]
        vec_W_action_traj[:, start_idx + 3 : start_idx + 9] = (
            my_vec_W_action_traj[:, 3:]
        )
    return vec_W_action_traj
