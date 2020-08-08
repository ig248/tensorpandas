import functools
import json
import operator

import numpy as np
import pandas.api.extensions as pdx
import pyarrow as pa

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
class TensorDtype(pdx.ExtensionDtype, metaclass=registry_type):
    name = "Tensor"
    kind = "O"
    type = np.ndarray
    na_value = np.nan

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        else:
            raise TypeError("Cannot construct a '{}' from " "'{}'".format(cls, string))

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


class TensorArray(pdx.ExtensionArray):
    ndim = 1

    def __init__(self, data):
        """Initialize from an nd-array or list of arrays."""
        if isinstance(data, self.__class__):
            self.data = data.data
            return
        try:
            self.data = np.stack(data)
        except ValueError as e:
            if isinstance(data, np.ndarray):
                self.data = data  # empty array
            else:
                raise ValueError("Incompatible data found at TensorArray initialization") from e

    # Attributes
    @property
    def dtype(self):
        return TensorDtype()

    @property
    def size(self):
        return len(self)

    def __len__(self):
        return self.data.shape[0]

    @property
    def tensor_shape(self):
        return self.data.shape

    @property
    def tensor_ndim(self):
        return self.data.ndim

    def __getitem__(self, idx):
        result = self.data[idx]
        if result.ndim < self.tensor_ndim:
            return result
        return self.__class__(result)

    # Methods
    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls(scalars)

    def isna(self):
        return np.any(np.isnan(self.data), axis=tuple(range(1, self.tensor_ndim)))

    def copy(self):
        return self.__class__(self.data.copy())

    def __array__(self, dtype=None):
        if dtype == np.dtype(object):
            # Return a 1D array for pd.array() compatibility
            return np.array([*self.data, None])[:-1]
        return self.data

    # Arithmetic methods
    def __eq__(self, other):
        return np.array_equal(self.data, other.data)

    # Arrow methods
    def __arrow_array__(self, type=None) -> pa.Array:
        # convert the underlying array values to a pyarrow Array
        subtype = pa.from_numpy_dtype(self.data.dtype)
        arrow_type = ArrowTensorType(shape=self.data.shape[1:], subtype=subtype)
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
        # verify there is a column latitude and a column longitude
        if not isinstance(obj.dtype, TensorDtype):
            raise AttributeError("Can only use .tensor accessor with Tensor values")

    @property
    def tensorarray(self):
        return self._obj.values

    @property
    def values(self):
        return self.tensorarray.data

    @property
    def dtype(self):
        return self.tensorarray.dtype

    @property
    def ndim(self):
        return self.tensorarray.tensor_ndim

    @property
    def shape(self):
        return self.tensorarray.tensor_shape
