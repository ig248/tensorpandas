import numpy as np
import pandas as pd
import pytest

from tensorpandas import TensorArray
from tensorpandas.base import _infer_na_value

n = 4


@pytest.fixture(params=[0, 1, 2])
def shape(request):
    return (2, 3)[: request.param]


@pytest.fixture(params=["float16", "float64", "datetime64[ns]"])
def dtype(request):
    return request.param


@pytest.fixture(params=["single_array", "sequence_of_arrays"])
def init_data_type(request):
    return request.param


@pytest.fixture
def init_data(init_data_type, shape, dtype):
    if init_data_type == "single_array":
        return np.random.rand(n, *shape).astype(dtype)
    elif init_data_type == "sequence_of_arrays":
        return [np.random.rand(*shape) for _ in range(n)]
    else:
        raise ValueError(f"Unknown data type: {init_data_type}")


@pytest.fixture
def ta(init_data):
    return TensorArray(init_data)


@pytest.fixture
def df(init_data, ta):
    df = pd.DataFrame(
        dict(
            x=np.arange(n),
            array=list(init_data),
            tensor=ta,
        )
    )
    return df


def test_tensor_array_values(shape, ta):
    assert ta._ndarray.shape == tuple([n, *shape])


def test_tensor_astype(ta, df):
    assert np.array_equal(df["array"].astype("Tensor"), ta)


def test_tensor_accessor(shape, ta, df):
    assert np.array_equal(df["tensor"].tensor.values, ta._ndarray)
    assert df["tensor"].tensor.ndim == len(shape)
    assert df["tensor"].tensor.shape == shape


def test_tensor_accessor_setter(shape, ta, df, dtype):
    if np.issubdtype(dtype, np.datetime64):
        pytest.skip()
    df["tensor"].tensor.values *= 0
    assert np.array_equiv(df["tensor"].tensor.values, 0)
    assert df["tensor"].tensor.ndim == len(shape)
    assert df["tensor"].tensor.shape == shape


def test_df_slice_concat(df):
    """Relies on _concat_same_type"""
    new_df = pd.concat([df.iloc[: n // 2], df.iloc[n // 2 :]])
    pd.testing.assert_frame_equal(new_df, df)


def test_df_iloc(df):
    df_iloc = df.iloc[:2]
    df_take = df.take([1, 0, -len(df) + 1]).take([-2, -1])
    pd.testing.assert_frame_equal(df_take, df_iloc)


def test_stack_unstack(init_data_type, df, dtype):
    if init_data_type != "single_array":
        pytest.skip()
    df = df[["tensor"]].copy()
    df["tensor2"] = df["tensor"]
    df.loc[df.index[-1], "tensor2"] = _infer_na_value(dtype)
    tall = df.stack()
    assert len(tall) == len(df) * 2 - 1
    assert tall.tensor.dtype == dtype
    wide = tall.unstack()
    assert len(wide) == len(df)
    assert wide["tensor"].tensor.dtype == dtype
    assert wide["tensor2"].tensor.dtype == dtype


@pytest.mark.parametrize("threshold", [0, 1, n])
def test_where(df, threshold):
    df, cond_df = df[["tensor"]], df[["x"]].rename(columns={"x": "tensor"})
    where_df = df[cond_df < threshold]
    assert where_df.shape == df.shape
    assert len(where_df.dropna(how="all", axis=0)) == threshold


def test_ufunc(shape, dtype, df):
    """Perform some basic arithmetic."""
    if np.issubdtype(dtype, np.datetime64):
        pytest.skip()
    # op with another TensorArray
    df["tensor"] -= df["tensor"]
    assert df["tensor"].tensor.shape == shape
    assert np.array_equiv(df["tensor"], 0)

    # op with another scalar
    df["tensor"] += 1
    assert df["tensor"].tensor.shape == shape
    assert np.array_equiv(df["tensor"], 1)

    # op with array (row-wise)
    arr = 2 * np.ones(shape)
    df["tensor"] += arr
    assert df["tensor"].tensor.shape == shape
    assert np.array_equiv(df["tensor"], 3)

    # op with array (column-wise)
    arr = np.arange(n).reshape(n, *(1 for _ in shape))
    df["tensor"] += arr
    assert df["tensor"].tensor.shape == shape
    assert np.array_equiv(df["tensor"], 3 + arr)
