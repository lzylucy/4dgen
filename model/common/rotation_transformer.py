import functools
from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from model.common.rotation_conversions import *


def get_rotation_transformer(from_rep: str, to_rep: str):
    assert from_rep == "axis_angle", f"Unsupported representation: {from_rep}"

    if to_rep == "rotation_6d":
        rotation_transformer = RotationTransformer(
            from_rep="axis_angle",
            to_rep=to_rep,
        )
    elif to_rep == "pitch":
        rotation_transformer = AxAng3ToPitch(
            from_rep="axis_angle",
            to_rep=to_rep,
        )
    else:
        assert False, f"Unsupported representation: {to_rep}"
    return rotation_transformer


class AxAng3ToPitch:
    def __init__(self, from_rep, to_rep):
        assert from_rep == "axis_angle"
        assert to_rep == "pitch"

    def forward(self, axang3):
        N, D = axang3.shape
        assert D == 3

        pitches = []
        for i in range(N):
            r = R.from_rotvec(axang3[i])
            zyx = r.as_euler("zyx")
            pitches.append([zyx[1]])
        pitches = np.array(pitches)
        assert pitches.shape == (N, 1)
        return pitches


class RotationTransformer:
    VALID_REPS = [
        "axis_angle",
        "euler_angles",
        "matrix",
        "quaternion",
        "rotation_6d",
    ]

    def __init__(
        self,
        from_rep="axis_angle",
        to_rep="rotation_6d",
        from_convention=None,
        to_convention=None,
    ):
        """
        Valid representations

        Alwasy use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.VALID_REPS
        assert to_rep in self.VALID_REPS
        if from_rep == "euler_angles":
            assert from_convention is not None
        if to_rep == "euler_angles":
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != "matrix":
            funcs = [
                globals()[f"{from_rep}_to_matrix"],
                globals()[f"matrix_to_{from_rep}"],
            ]
            if from_convention is not None:
                funcs = [
                    functools.partial(func, convernsion=from_convention)
                    for func in funcs
                ]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = [
                globals()[f"matrix_to_{to_rep}"],
                globals()[f"{to_rep}_to_matrix"],
            ]
            if to_convention is not None:
                funcs = [
                    functools.partial(func, convernsion=to_convention)
                    for func in funcs
                ]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    def forward(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        return _apply_funcs(x, self.forward_funcs)

    def inverse(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        return _apply_funcs(x, self.inverse_funcs)


def _apply_funcs(
    x: Union[np.ndarray, torch.Tensor], funcs: list
) -> Union[np.ndarray, torch.Tensor]:

    is_ndarray = isinstance(x, np.ndarray)
    if is_ndarray:
        x = torch.from_numpy(x)
    else:
        assert isinstance(x, torch.Tensor), f"Unknown data type: {type(x)}"

    for func in funcs:
        x = func(x)

    if is_ndarray:
        x = x.numpy()
    return x


def test():
    tf = RotationTransformer()

    rotvec = np.random.uniform(-2 * np.pi, 2 * np.pi, size=(1000, 3))
    rot6d = tf.forward(rotvec)
    new_rotvec = tf.inverse(rot6d)

    from scipy.spatial.transform import Rotation

    diff = (
        Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(new_rotvec).inv()
    )
    dist = diff.magnitude()
    assert dist.max() < 1e-7

    tf = RotationTransformer("rotation_6d", "matrix")
    rot6d_wrong = rot6d + np.random.normal(scale=0.1, size=rot6d.shape)
    mat = tf.forward(rot6d_wrong)
    mat_det = np.linalg.det(mat)
    assert np.allclose(mat_det, 1)
    # rotaiton_6d will be normalized to rotation matrix
