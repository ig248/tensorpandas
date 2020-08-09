import numpy as np
import pandas as pd
import pytest

from tensorpandas import TensorArray

n = 100


@pytest.fixture(params=[0, 1, 2])
def shape(request):
    return (2, 3)[: request.param]


@pytest.fixture(params=["single_array", "sequence_of_arrays"])
def init_data_type(request):
    return request.param


@pytest.fixture
def init_data(init_data_type, shape):
    if init_data_type == "single_array":
        return np.random.rand(n, *shape)
    elif init_data_type == "sequence_of_arrays":
        return [np.random.rand(*shape) for _ in range(n)]
    else:
        raise ValueError(f"Unknown data type: {init_data_type}")


@pytest.fixture
def ta(init_data):
    return TensorArray(init_data)


@pytest.fixture
def df(init_data, ta):
    df = pd.DataFrame(dict(x=np.arange(n), array=list(init_data), tensor=ta,))
    return df


def test_tensor_array_values(shape, ta):
    assert ta._ndarray.shape == tuple([n, *shape])


def test_tensor_astype(ta, df):
    assert all(df["array"].astype("Tensor") == ta)


def test_tensor_accessor(shape, ta, df):
    assert np.array_equal(df["tensor"].tensor.values, ta._ndarray)
    assert df["tensor"].tensor.ndim == len(shape) + 1
    assert df["tensor"].tensor.shape == (n, *shape)


def test_df_slice_concat(df):
    """Relies on _concat_same_type"""
    new_df = pd.concat([df.iloc[: n // 2], df.iloc[n // 2 :]])
    pd.testing.assert_frame_equal(new_df, df)


def test_df_iloc(df):
    df_iloc = df.iloc[:2]
    df_take = df.take([1, 0, -len(df) + 1]).take([-2, -1])
    pd.testing.assert_frame_equal(df_take, df_iloc)
