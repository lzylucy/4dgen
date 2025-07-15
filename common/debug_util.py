"""
Utilities that should be synchronized with:
https://drake.mit.edu/python_bindings.html#debugging-with-the-python-bindings
"""

from contextlib import contextmanager
import pdb
import sys
import traceback


def reexecute_if_unbuffered():
    """Ensures that output is immediately flushed (e.g. for segfaults).
    ONLY use this at your entrypoint. Otherwise, you may have code be
    re-executed that will clutter your console."""
    import os
    import shlex
    import sys

    if os.environ.get("PYTHONUNBUFFERED") in (None, ""):
        os.environ["PYTHONUNBUFFERED"] = "1"
        argv = list(sys.argv)
        if argv[0] != sys.executable:
            argv.insert(0, sys.executable)
        cmd = " ".join([shlex.quote(arg) for arg in argv])
        sys.stdout.flush()
        os.execv(argv[0], argv)


def traced(func, ignoredirs=None):
    """Decorates func such that its execution is traced, but filters out any
    Python code outside of the system prefix."""
    import functools
    import sys
    import trace

    if ignoredirs is None:
        ignoredirs = ["/usr", sys.prefix]
    tracer = trace.Trace(trace=1, count=0, ignoredirs=ignoredirs)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return tracer.runfunc(func, *args, **kwargs)

    return wrapped


@contextmanager
def launch_pdb_on_exception():
    """
    Provides a context that will launch interactive pdb console automatically
    if an exception is raised.

    Example usage with @iex decorator below:

        @iex
        def my_bad_function():
            x = 1
            assert False

        my_bad_function()
        # Should bring up debugger at `assert` statement.
    """
    # Adapted from:
    # https://github.com/gotcha/ipdb/blob/fc83b4f5f/ipdb/__main__.py#L219-L232

    try:
        yield
    except (Exception, SystemExit):
        traceback.print_exc()
        _, _, tb = sys.exc_info()
        pdb.post_mortem(tb)
        # Resume original execution.
        raise


# See docs for `launch_pdb_on_exception()`.
iex = launch_pdb_on_exception()
