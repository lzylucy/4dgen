import logging
import os
import sys


def _workaround():
    if "transformers" in sys.modules:
        raise RuntimeError(
            "This module must be imported *before* transformers"
        )

    if "TEST_TMPDIR" in os.environ:
        # Prevent from cookies being saved to the home directory.
        import gdown

        _old_download = gdown.download

        def _new_download(*args, **kwargs):
            kwargs["use_cookies"] = False
            return _old_download(*args, **kwargs)

        gdown.download = _new_download

        # Change cache directory.
        os.environ["TRANSFORMERS_CACHE"] = os.environ["TEST_TMPDIR"]
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["TEST_TMPDIR"]

    # Since transformers can't recognize our PyTorch and torchvision,
    # we hard-code this here.

    class TransformersWarningFilter(logging.Filter):
        def filter(self, record):
            # This warning message is sent from transformers.__init__.py
            message = (
                "None of PyTorch, TensorFlow >= 2.0, or Flax have been found. "
                "Models won't be available and only tokenizers, configuration "
                "and file/data utilities can be used."
            )
            if record.getMessage() == message:
                record.msg = ""
            return True

    logger = logging.getLogger("transformers")
    handler = logging.StreamHandler()
    handler.addFilter(TransformersWarningFilter())
    logger.addHandler(handler)

    import transformers  # noqa: transformers_pre_import_mods

    transformers.utils.import_utils._torch_available = True
    transformers.utils.import_utils._torch_version = "2.0.0"
    transformers.utils.import_utils._torch_fx_available = True

    def _is_available_always_true():
        return True

    transformers.is_torch_available = _is_available_always_true
    transformers.is_torch_vision_available = _is_available_always_true

    import importlib

    importlib.reload(transformers)


# N.B. We don't care if this is imported by the main module or a library.
assert __name__ != "__main__"
_workaround()
