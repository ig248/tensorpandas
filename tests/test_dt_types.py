import numpy as np
import pandas as pd
import pytest

from tensorpandas import TensorArray

n = 4


@pytest.fixture(params=[1, 2])
def shape(request):
    return (2, 3)[: request.param]


@pytest.fixture(params=["int", "float", "datetime64[ms]"])
def dtype(request):
    return request.param


@pytest.fixture
def init_data(shape, dtype):
    return np.random.rand(n, *shape).astype(dtype)


@pytest.fixture
def ta(init_data):
    return TensorArray(init_data)


@pytest.fixture
def series(ta):
    return pd.Series(ta)


@pytest.fixture
def df(series):
    return pd.DataFrame({"col": series})


def test_repr(ta, series, df):
    for obj in [ta, series, df]:
        _ = repr(obj)


def test_str(ta, series, df):
    for obj in [ta, series, df]:
        _ = str(obj)
