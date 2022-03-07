import pandas
from packaging import version

ver = version.parse(pandas.__version__)
ver_exc = NotImplementedError(f"pandas == {pandas.__version__} is not supported")


if ver < version.parse("1.1"):
    raise ver_exc
elif ver < version.parse("1.2"):
    from . import _patch_1_1  # noqa
elif ver < version.parse("1.3"):
    from . import _patch_1_2  # noqa
elif ver < version.parse("1.4"):
    from . import _patch_1_3  # noqa
else:
    raise ver_exc
