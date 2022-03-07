from __future__ import annotations

import numpy as np
from pandas.core.dtypes.common import is_list_like

from .base import TensorArray


# patch pandas.core.arrays.base.ExtensionArray._where
def _where(self: TensorArray, mask, value) -> TensorArray:
    """
    Analogue to np.where(mask, self, value)

    Parameters
    ----------
    mask : np.ndarray[bool]
    value : scalar or listlike

    Returns
    -------
    same type as self
    """
    result = self.copy()
    result[~mask] = value
    return result


TensorArray._where = _where


# patch format
from pandas.io.formats import format


# formatting now goes through ExtensionArrayFormatter regardless of dtype (incl. datetimes)
def _format_strings(self) -> list[str]:
    values = format.extract_array(self.values, extract_numpy=True)

    formatter = self.formatter
    if formatter is None:
        # error: Item "ndarray" of "Union[Any, Union[ExtensionArray, ndarray]]" has
        # no attribute "_formatter"
        formatter = values._formatter(boxed=True)  # type: ignore[union-attr]

    if isinstance(values, format.Categorical):
        # Categorical is special for now, so that we can preserve tzinfo
        array = values._internal_get_values()
    elif values.dtype.is_dtype("Tensor"):
        # transform to 1D object array
        array = np.empty((len(values),), dtype=object)
        array[:] = list(values)
    else:
        array = np.asarray(values)

    fmt_values = format.format_array(
        array,
        formatter,
        float_format=self.float_format,
        na_rep=self.na_rep,
        digits=self.digits,
        space=self.space,
        justify=self.justify,
        decimal=self.decimal,
        leading_space=self.leading_space,
        quoting=self.quoting,
    )
    return fmt_values


format.ExtensionArrayFormatter._format_strings = _format_strings
