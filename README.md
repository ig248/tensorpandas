# tensorpandas

Provides extension types to store arbitrary ndarrays (tensors) in pandas objects:

```python
n = 100
shape = (2, 3)
arrays = [np.random.rand(*shape) for _ in range(n)]

df = pd.DataFrame({
    "x": np.arange(n),
    "tensor": TensorArray(arrays)
))

df["same_tensor"] = pd.Series(arrays).astype("Tensor")  # also works

df["tensor"].tensor.values  # shape (100, 2, 3) for direct data access
df.to_parquet("test.parquet". engine="pyarrow")  # store each tensor as fixed-size binary blob
```

## Features
- store n-dimensional arrays in a pandas Series or DataFrame
- support for `pyarrow`: `df.to_parquet`/`pd.read_parquet` support the Tensor extension type
- data is accessable as an (n+1)-dimensional array for efficient slicing or in-place manipulation
