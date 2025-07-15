import numpy as np
import scipy.spatial.transform as st


def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=pos.dtype)
    mat[..., :3, 3] = pos
    mat[..., :3, :3] = rot.as_matrix()
    mat[..., 3, 3] = 1
    return mat


def mat_to_pos_rot(mat):
    pos = (mat[..., :3, 3].T / mat[..., 3, 3].T).T
    rot = st.Rotation.from_matrix(mat[..., :3, :3])
    return pos, rot


def pos_rot_to_pose(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros(shape + (6,), dtype=pos.dtype)
    pose[..., :3] = pos
    pose[..., 3:] = rot.as_rotvec()
    return pose


def pose_to_pos_rot(pose):
    pos = pose[..., :3]
    rot = st.Rotation.from_rotvec(pose[..., 3:])
    return pos, rot


def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))


def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))


def transform_pose(tx, pose):
    """
    tx: tx_new_old
    pose: tx_old_obj
    result: tx_new_obj
    """
    pose_mat = pose_to_mat(pose)
    tf_pose_mat = tx @ pose_mat
    tf_pose = mat_to_pose(tf_pose_mat)
    return tf_pose


def transform_point(tx, point):
    return point @ tx[:3, :3].T + tx[:3, 3]


def project_point(k, point):
    x = point @ k.T
    uv = x[..., :2] / x[..., [2]]
    return uv


def apply_delta_pose(pose, delta_pose):
    new_pose = np.zeros_like(pose)

    # simple add for position
    new_pose[:3] = pose[:3] + delta_pose[:3]

    # matrix multiplication for rotation
    rot = st.Rotation.from_rotvec(pose[3:])
    drot = st.Rotation.from_rotvec(delta_pose[3:])
    new_pose[3:] = (drot * rot).as_rotvec()

    return new_pose


def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out = (vec.T / norm).T
    return out


def rot6d_to_mat(d6):
    """
    Takes a [N, 6] np array and convert to [N, 3, 3] rotation matrices.
    The input is assumed to be the first 2 rows of a rotation matrix.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out


def mat_to_rot6d(mat):
    """
    Takes a [N, 3, 3] np array and convert to [N, 6] array.
    Inverse of rot6d_to_mat().
    """
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out


def mat_to_pose9d(mat):
    """
    Takes a [N, 4, 4] np array of transformations and convert to [N, 9].
    Inverse of pose9d_to_mat()
    """
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    d6 = mat_to_rot6d(rotmat)
    d9 = np.concatenate([pos, d6], axis=-1)
    return d9


def pose9d_to_mat(d9):
    """
    Takes a [N, 9] np array and convert to [N, 4, 4] transformations.
    Each row of input is assumed to be xyz, 6d rotation.
    """
    pos = d9[..., :3]
    d6 = d9[..., 3:]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d9.shape[:-1] + (4, 4), dtype=d9.dtype)
    out[..., :3, :3] = rotmat
    out[..., :3, 3] = pos
    out[..., 3, 3] = 1
    return out


def convert_pose_mat_rep(pose_mat, X_A_C, pose_rep="abs", backward=False):
    """
    Convert the [N, 4, 4] `pose_mat` pose trajectory depending on `pose_rep`,
    which can be "abs" or "relative".
    In "abs" mode:
        - `pose_mat` is directly returned.
    In "relative" mode:
        - when "backward" is False, `pose_mat` is treated as X_A_B_traj,
            and this func returns X_C_B_traj.
        - when "backward" is True, `pose_mat` is treated as X_C_B_traj,
            and this func returns X_A_B_traj.

    A concrete example for converting a trajectory of ee pose in world frame
    (X_W_EE) to a trajectory of ee pose wrt some other frame in world (X_W_F)
    X_F_EE = convert_pose_mat_rep(
        X_W_EE, X_W_F, pose_rep="relative", backward=False
    )
    To convert X_F_EE to X_W_EE again:
    X_W_EE = convert_pose_mat_rep(
        X_F_EE, X_W_F, pose_rep="relative", backward=True
    )
    """
    if not backward:
        # training transform
        if pose_rep == "abs":
            return pose_mat
        elif pose_rep == "relative":
            out = np.linalg.inv(X_A_C) @ pose_mat
            return out
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")

    else:
        # eval transform
        if pose_rep == "abs":
            return pose_mat
        elif pose_rep == "relative":
            out = X_A_C @ pose_mat
            return out
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")
