import pandas._libs.lib
from pandas.io.formats import format
from pandas.io.formats.format import *
from pandas.io.formats.format import _get_format_datetime64_from_values

from .base import NaArray


# This fixes casting issues BlockManager.where()
_original_is_scalar = pandas._libs.lib.is_scalar


def is_scalar(val: object) -> bool:
    if isinstance(val, NaArray):
        return True
    return _original_is_scalar(val)


pandas._libs.lib.is_scalar = is_scalar



# This fixes issues with repr for tensor arrays of datetimes
class Datetime64Formatter(GenericArrayFormatter):
    def __init__(
        self,
        values: Union[np.ndarray, "Series", DatetimeIndex, DatetimeArray],
        nat_rep: str = "NaT",
        date_format: None = None,
        **kwargs,
    ):
        super().__init__(values, **kwargs)
        self.nat_rep = nat_rep
        self.date_format = date_format

    def _format_strings(self) -> List[str]:
        """ we by definition have DO NOT have a TZ """

        values = self.values

        # hacky patch to avoid display errors
        if not isinstance(values, DatetimeIndex) and len(values.shape) == 1:
            values = DatetimeIndex(values)
        if self.formatter is not None and callable(self.formatter):
            return [self.formatter(x) for x in values]

        fmt_values = format_array_from_datetime(
            values.asi8.ravel(),
            format=_get_format_datetime64_from_values(values, self.date_format),
            na_rep=self.nat_rep,
        ).reshape(values.shape)
        return fmt_values.tolist()


format.Datetime64Formatter = Datetime64Formatter
