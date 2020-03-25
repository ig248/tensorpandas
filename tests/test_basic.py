import numpy as np
import pandas as pd

from tensorpandas import TensorArray


n = 100
shape = (2, 3)
arrays = [np.random.rand(*shape) for _ in range(n)]
ta = TensorArray(arrays)
df = pd.DataFrame(dict(
    x=np.arange(n),
    array=arrays,
    tensor=ta,
))


def test_tensor_array_values():
    assert ta.data.shape == tuple([n, *shape])


def test_tensor_astype():
    assert all(df["array"].astype("Tensor") == ta)


def test_tensor_accessor():
    assert np.array_equal(df["tensor"].tensor.values, ta.data)
