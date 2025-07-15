from collections import UserDict
from typing import Any, Dict, Union

import fsspec
import numpy as np


def load_file_with_fsspec(path: str, loader: Any, mode: str = "r") -> Any:
    """
    Load a file from `path`. The location of the file can be either on
    local, including locally mounted EFS, or S3.
    """
    if path.endswith(".npz"):
        raise ValueError("Use load_npz_with_fsspec() instead.")

    fs, fsspec_path = fsspec.core.url_to_fs(path)
    with fs.open(fsspec_path, mode) as f:
        contents = loader(f)
    return contents


def load_npz_with_fsspec(
    path: str,
    lazy_load: bool = True,
) -> Union[UserDict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load a npz file from `path`. The location of the npz file can be either on
    local, including locally mounted EFS, or S3. It lazily loads the inner
    files if `lazy_load` is True (default.)
    """
    assert path.endswith(".npz"), f"Found non-npz file: {path}."
    fs, fsspec_path = fsspec.core.url_to_fs(path)

    with fs.open(fsspec_path, "rb") as f:
        contents = np.load(f)
        keys = contents.keys()

        if not lazy_load:
            output = dict()
            for key in keys:
                output[key] = contents[key]

            return output

    class LazyDict(UserDict):

        def __getitem__(self, key):
            with fs.open(fsspec_path, "rb") as f:
                contents = np.load(f)
                return contents[key]

    output = LazyDict()
    output.update(dict.fromkeys(keys, None))
    return output
