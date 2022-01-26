from packaging import version
import pandas


ver = version.parse(pandas.__version__)
ver_exc = NotImplementedError(f"{pandas.__version__=} is not supported")


if ver < version.parse("1.1"):
    raise ver_exc
elif ver < version.parse("1.2"):
    from . import _patch_1_1
else:
    raise ver_exc
