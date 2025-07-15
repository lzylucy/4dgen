import functools
import glob
import itertools
import os
from os.path import abspath, expanduser, expandvars
from pathlib import Path
import re

_MARKER = object()


@functools.cache
def check_for_runfiles():
    try:
        from bazel_tools.tools.python.runfiles.runfiles import (
            Create as CreateRunfiles,
        )

        return CreateRunfiles()
    except ImportError:
        return None


def interleave_longest(iters):
    """
    Interleaves elements in each iterator in ``iters``, which may produce
    sequences of different lengths.

    Derived from:'
    - https://stackoverflow.com/a/40954220/7829525
    - https://more-itertools.readthedocs.io/en/v8.12.0/_modules/more_itertools/more.html#interleave_longest
    """  # noqa
    zip_iter = itertools.zip_longest(*iters, fillvalue=_MARKER)
    for x in itertools.chain(*zip_iter):
        if x is not _MARKER:
            yield x


def resolve_path(path):
    """
    Expands a path, expanding variables, ~, and package URLs. To parse a path
    with `package://`, we will use sys.path as an alternative to support the
    non-Bazel workflow. For the path with `s3://`, it does nothing.
    """
    pkg_prefix = "package://"
    if path.startswith(pkg_prefix):
        pkg_suffix = path[len(pkg_prefix) :]
        if (runfiles := check_for_runfiles()) is not None:
            return runfiles.Rlocation(pkg_suffix)
        lbm_root = Path(__file__).parent.parent.parent
        assert pkg_suffix.startswith("lbm/"), path
        return lbm_root / pkg_suffix[4:]
    elif path.startswith("s3://"):
        # Do nothing.
        return path
    else:
        path = expandvars(expanduser(path))
        return abspath(path)


def _tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def human_sorting_key(text):
    return tuple(_tryint(s) for s in re.split(r"(\d+)", text))


def human_sorted_strings(items):
    """
    Applies human sorting to a list of strings.

    See also: https://nedbatchelder.com/blog/200712/human_sorting.html
    """
    assert isinstance(items, list)
    return list(sorted(items, key=human_sorting_key))


def resolve_glob_type_to_list(
    globs,
    *,
    allow_empty=False,
    sort=True,
    interleave=False,
):
    if interleave:
        assert sort, "Can only interleave if sorting"
    if isinstance(globs, str):
        globs = [globs]
    assert isinstance(globs, list)
    files_per_glob = []
    for glob_str in globs:
        glob_str = expandvars(expanduser(glob_str))
        glob_files = glob.glob(glob_str)
        if not allow_empty:
            assert len(glob_files) > 0, repr(glob_str)
        if interleave:
            # Sort result from each glob.
            glob_files = human_sorted_strings(glob_files)
        files_per_glob.append(glob_files)
    if sort:
        if interleave:
            files = list(interleave_longest(files_per_glob))
        else:
            files = []
            for files_i in files_per_glob:
                files += files_i
            files = human_sorted_strings(files)
    return files


def abs_efs_path_to_tilde_path(path):
    parts = path.split(os.sep)
    assert "efs" in parts, f"{path} not valid efs path"

    index = parts.index("efs")
    parts = ["~"] + parts[index:]

    return os.sep.join(parts)


def resolve_efs_path(path):
    path = abs_efs_path_to_tilde_path(path)
    return resolve_path(path)


def Rlocation(respath):
    runfiles = check_for_runfiles()
    assert runfiles is not None, "Only usable in Bazel"
    path = runfiles.Rlocation(respath)
    assert path is not None, respath
    return path


def rules_python_entry_point(pkg, script=None):
    """Runtime workalike of rules_python entry_point macro:  Given a python
    package name and the name of a defined entry point of that package, get
    the corresponding bazel rlocation."""
    if not script:
        script = pkg
    script_prefix = "rules_python_wheel_entry_point"
    return rules_python_respath(pkg, f"{script_prefix}_{script}")


def rules_python_respath(pkg, relpath):
    """Runtime respath of a file contained under a rules_python package."""
    our_pkg_prefix = "requirements"
    external = f"{our_pkg_prefix}_{pkg}"
    return f"{external}/{relpath}"
