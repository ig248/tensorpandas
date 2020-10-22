import functools
import json
import numbers
import operator
from typing import Any, Dict, Sequence, Union

import numpy as np
import pandas.api.extensions as pdx
import pyarrow as pa
from numpy.lib.mixins import NDArrayOperatorsMixin
from pandas._libs import lib
from pandas.core.arrays import PandasArray
from pandas.core.indexers import check_array_indexer
from pandas.core.dtypes.dtypes import PandasExtensionDtype
from pandas.core.dtypes.common import pandas_dtype


__all__ = ["TensorDtype", "TensorArray"]


# https://arrow.apache.org/docs/python/extending_types.html#parametrized-extension-type
class ArrowTensorType(pa.ExtensionType):
    def __init__(self, shape, subtype):
        # attributes need to be set first before calling
        # super init (as that calls serialize)
        self._shape = shape
        self._subtype = subtype
        if not isinstance(subtype, pa.DataType):
            subtype = pa.type_for_alias(str(subtype))
        size = functools.reduce(operator.mul, shape)
        self._storage_type = pa.binary(size * subtype.bit_width // 8)
        pa.ExtensionType.__init__(self, self._storage_type, "tensorpandas.tensor")

    @property
    def bit_width(self):
        return self._storage_type.bit_width

    @property
    def shape(self):
        return tuple(self._shape)

    @property
    def subtype(self):
        return self._subtype

    def __arrow_ext_serialize__(self):
        metadata = {"shape": self.shape, "subtype": str(self.subtype)}
        return json.dumps(metadata).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        # return an instance of this subclass given the serialized
        # metadata.
        metadata = json.loads(serialized.decode())
        shape = metadata["shape"]
        subtype = pa.type_for_alias(metadata["subtype"])
        return ArrowTensorType(shape=shape, subtype=subtype)

    def to_pandas_dtype(self):
        return TensorDtype()


# register the type with a dummy instance
_tensor_type = ArrowTensorType((1,), pa.float32())
pa.register_extension_type(_tensor_type)


class registry_type(type):
    """Fix registry lookup for extension types.

    It appears that parquet stores `str(TensorDtype)`, yet the
    lookup tries to match it to `TensorDtype.name`.
    """

    def __str__(self):
        try:
            return self.name
        except AttributeError:
            return self.__name__


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.extensions.ExtensionDtype.html
@pdx.register_extension_dtype
class TensorDtype(PandasExtensionDtype, metaclass=registry_type):
    # kind = "O"
    type = np.ndarray
    _metadata = ("shape", "_dtype")
    _cache: Dict[tuple, "TensorDtype"] = {}

    def __new__(cls, shape=(), dtype=None):
        if not isinstance(shape, tuple):
            raise TypeError("Shape must be a tuple")
        try:
            dtype = np.dtype(dtype)
        except TypeError as err:
            raise ValueError(f"{dtype} is not a valid dtype") from err
        if (shape, dtype) not in cls._cache:
            cls._cache[(shape, dtype)] = super().__new__(cls)
        return cls._cache[(shape, dtype)]

    def __init__(self, shape=(), dtype=None):
        self.shape = shape
        # we can not use .dtype as it is leads to conflicts in e.g. is_extension_array_dtype
        self._dtype = np.dtype(dtype)

    @classmethod
    def construct_from_string(cls, string):
        if string == "Tensor":
            return cls()
        try:
            return eval(string, {}, {"Tensor": cls, "dtype": np.dtype})
        except Exception as err:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'") from err

    @property
    def na_value(self):
        na = np.nan + np.empty(self.shape, dtype=self._dtype)
        return na

    def __str__(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        return f"Tensor(shape={self.shape!r}, dtype={self._dtype!r})"

    def __hash__(self) -> int:
        # make myself hashable
        return hash(str(self))

    @classmethod
    def construct_array_type(cls):
        return TensorArray

    def __from_arrow__(self, array) -> pdx.ExtensionArray:
        """Construct TensorArray from pyarrow Array/ChunkedArray."""
        if isinstance(array, pa.Array):
            chunks = [array]
        else:
            chunks = array.chunks
        tensors = []
        for arr in chunks:
            shape = arr.type.shape
            subtype = arr.type.subtype
            for tensor in arr.storage.to_numpy(zero_copy_only=False):
                tensors.append(
                    np.frombuffer(tensor, dtype=subtype.to_pandas_dtype()).reshape(shape)
                )
        return TensorArray(np.stack(tensors))


class TensorArray(pdx.ExtensionArray, NDArrayOperatorsMixin):
    ndim = 1

    def __init__(self, data):
        """Initialize from an nd-array or list of arrays."""
        if isinstance(data, self.__class__):
            self._ndarray = data._ndarray
        elif isinstance(data, np.ndarray) and data.dtype != object:  # i.e. not array of arrays
            self._ndarray = data
        else:
            self._ndarray = np.stack(data)

    # Attributes
    @property
    def dtype(self):
        return TensorDtype(shape=self.tensor_shape, dtype=self.tensor_dtype)

    @property
    def size(self):
        return len(self)

    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.
        """
        return self._ndarray.nbytes

    def __len__(self):
        return self._ndarray.shape[0]

    @property
    def tensor_shape(self):
        return self._ndarray.shape[1:]

    @property
    def tensor_ndim(self):
        return self._ndarray.ndim - 1

    @property
    def tensor_dtype(self):
        return self._ndarray.dtype

    def __getitem__(self, item):
        if isinstance(item, type(self)):
            item = item._ndarray
        item = check_array_indexer(self, item)
        result = self._ndarray[item]
        if result.ndim < self._ndarray.ndim:
            return result
        return self.__class__(result)

    def __setitem__(self, key: Union[int, np.ndarray], value: Any) -> None:
        """
        Set one or more values inplace.

        Parameters
        ----------
        key : int, ndarray, or slice
            When called from, e.g. ``Series.__setitem__``, ``key`` will be
            one of

            * scalar int
            * ndarray of integers.
            * boolean ndarray
            * slice object

        value : ExtensionDtype.type, Sequence[ExtensionDtype.type], or object
            value or values to be set of ``key``.

        Returns
        -------
        None
        """
        self._ndarray[key] = value

    # Methods
    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls(scalars)

    @classmethod
    def _concat_same_type(cls, to_concat):
        return cls(np.concatenate([arr._ndarray for arr in to_concat]))

    def astype(self, dtype, copy=True):
        """
        Cast to an array with 'dtype'. Currently only support conversion
        to same Tensor type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : TensorArray
        """

        dtype = pandas_dtype(dtype)
        if isinstance(dtype, TensorDtype):
            return self.copy() if copy else self
        return np.array(self, dtype=dtype, copy=copy)

    def isna(self):
        return np.any(np.isnan(self._ndarray), axis=tuple(range(1, self._ndarray.ndim)))

    def take(
        self, indices: Sequence[int], allow_fill: bool = False, fill_value: Any = None
    ) -> "TensorArray":
        """
        Take elements from an array.

        Parameters
        ----------
        indices : sequence of int
            Indices to be taken.
        allow_fill : bool, default False
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.

            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.

        fill_value : any, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.

            For many ExtensionArrays, there will be two representations of
            `fill_value`: a user-facing "boxed" scalar, and a low-level
            physical NA value. `fill_value` should be the user-facing version,
            and the implementation should handle translating that to the
            physical version for processing the take if necessary.

        Returns
        -------
        ExtensionArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.

        See Also
        --------
        numpy.take
        api.extensions.take
        """
        if fill_value is None:
            fill_value = self.dtype.na_value
        _result = fill_value + np.zeros(
            (len(indices), *self.tensor_shape), dtype=self.dtype._dtype
        )
        if allow_fill:
            indices = np.array(indices)
            if np.any((indices < 0) & (indices != -1)):
                raise ValueError("Fill points must be indicated by -1")
            destination = indices >= 0  # boolean
            indices = indices[indices >= 0]
        else:
            destination = slice(None, None, None)
        if len(indices) > 0 and not self._ndarray.shape[0]:
            raise IndexError("cannot do a non-empty take")
        _result[destination] = self._ndarray[indices]

        return self.__class__(_result)

    def copy(self):
        return self.__class__(self._ndarray.copy())

    def view(self):
        return self.__class__(self._ndarray)

    def __array__(self, dtype=None):
        if dtype == np.dtype(object):
            # Return a 1D array for pd.array() compatibility
            return np.array([*self._ndarray, None], dtype=object)[:-1]
        return self._ndarray

    # adopted from PandasArray
    _HANDLED_TYPES = (np.ndarray, numbers.Number, PandasArray)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get("out", ())
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (self.__class__,)):
                return NotImplemented

        # Ther doesn't seem to be a way of knowing if another array came from a PandasArray
        # This creates a huge confusion between column and row arrays.
        def _as_array(x):
            if isinstance(x, self.__class__):
                x = x._ndarray
            return x

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(_as_array(x) for x in inputs)
        if out:
            kwargs["out"] = tuple(_as_array(x) for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple and len(result):
            # multiple return values
            if not lib.is_scalar(result[0]):
                # re-box array-like results
                return tuple(type(self)(x) for x in result)
            else:
                # but not scalar reductions
                return result
        elif method == "at":
            # no return value
            return None
        else:
            # one return value
            if not lib.is_scalar(result):
                # re-box array-like results, but not scalar reductions
                result = type(self)(result)
            return result

    # Arrow methods
    def __arrow_array__(self, type=None) -> pa.Array:
        # convert the underlying array values to a pyarrow Array
        subtype = pa.from_numpy_dtype(self._ndarray.dtype)
        arrow_type = ArrowTensorType(shape=self._ndarray.shape[1:], subtype=subtype)
        storage_array = pa.array(
            [item.tobytes() for item in self], type=arrow_type._storage_type, from_pandas=True
        )
        return pa.ExtensionArray.from_storage(arrow_type, storage_array)


@pdx.register_series_accessor("tensor")
class TensorAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if not isinstance(obj.dtype, TensorDtype):
            raise AttributeError("Can only use .tensor accessor with Tensor values")

    @property
    def tensorarray(self):
        return self._obj.values

    @property
    def values(self):
        return self.tensorarray._ndarray

    @values.setter
    def values(self, new_values):
        self.tensorarray._ndarray = new_values

    @property
    def dtype(self):
        return self.tensorarray.tensor_dtype

    @property
    def ndim(self):
        return self.tensorarray.tensor_ndim

    @property
    def shape(self):
        return self.tensorarray.tensor_shape
