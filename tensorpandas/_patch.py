## NB: this works with 1.0.5 but is a terrible hack...
# patch block
import numpy as np

from pandas._libs import lib
from pandas.core.dtypes.generic import ABCDataFrame
from pandas.core.construction import extract_array
from pandas.core.dtypes.missing import isna
from pandas.core.dtypes.common import is_sparse

import pandas.core.internals
# patch format
from pandas.io.formats import format
from pandas.io.formats.format import *
from pandas.io.formats.format import _get_format_datetime64_from_values


# This fixes casting issues BlockManager.where()
def where(
    self,
    other,
    cond,
    align=True,
    errors="raise",
    try_cast: bool = False,
    axis: int = 0,
):
    if isinstance(other, ABCDataFrame):
        # ExtensionArrays are 1-D, so if we get here then
        # `other` should be a DataFrame with a single column.
        assert other.shape[1] == 1
        other = other.iloc[:, 0]

    other = extract_array(other, extract_numpy=True)

    if isinstance(cond, ABCDataFrame):
        assert cond.shape[1] == 1
        cond = cond.iloc[:, 0]

    cond = extract_array(cond, extract_numpy=True)

    if lib.is_scalar(other) and isna(other):
        # The default `other` for Series / Frame is np.nan
        # we want to replace that with the correct NA value
        # for the type
        other = self.dtype.na_value

    if is_sparse(self.values):
        # TODO(SparseArray.__setitem__): remove this if condition
        # We need to re-infer the type of the data after doing the
        # where, for cases where the subtypes don't match
        dtype = None
    else:
        dtype = self.dtype

    result = self.values.copy()
    icond = ~cond
    if lib.is_scalar(other) or self.dtype.is_dtype("Tensor"):
        set_other = other
    else:
        set_other = other[icond]
    try:
        result[icond] = set_other
    except (NotImplementedError, TypeError):
        # NotImplementedError for class not implementing `__setitem__`
        # TypeError for SparseArray, which implements just to raise
        # a TypeError
        result = self._holder._from_sequence(
            np.where(cond, self.values, other), dtype=dtype
        )

    return [self.make_block_same_class(result, placement=self.mgr_locs)]


pandas.core.internals.blocks.ExtensionBlock.where = where


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
