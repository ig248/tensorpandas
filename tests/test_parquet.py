import pandas as pd


def test_parquet_roundtrip(data, tmp_path):
    filename = tmp_path / "test.parquet"
    df = pd.DataFrame({"x": range(len(data)), "tensors": data})
    df.to_parquet(filename, engine="pyarrow")
    df2 = pd.read_parquet(filename)

    assert "tensors" in df2
    assert all(df["tensors"] == df2["tensors"])
