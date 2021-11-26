import pytest
from pandas.core.dtypes.common import is_extension_array_dtype

from tensorpandas import TensorDtype


@pytest.fixture(params=[(), (64, 64)])
def shape(request):
    return request.param


@pytest.fixture(params=[None, "int", "float16"])
def dtype(request):
    return request.param


def test_from_string_roundtrip(shape, dtype):
    tdtype = TensorDtype(shape=shape, dtype=dtype)
    assert TensorDtype.construct_from_string(tdtype.name) is tdtype


def test_from_string(shape, dtype):
    assert TensorDtype.construct_from_string("Tensor") is TensorDtype()


def test_extension_type_resolution():
    assert not is_extension_array_dtype("str")
    assert not is_extension_array_dtype("object")
