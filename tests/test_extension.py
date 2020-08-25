import pytest
import operator

import numpy as np
import pandas as pd
from pandas.tests.extension import base
import pandas._testing as tm


class TestDtype(base.BaseDtypeTests):
    pass


class TestInterface(base.BaseInterfaceTests):
    def test_array_interface(self, data):
        result = np.array(data)
        tm.assert_numpy_array_equal(result[0], data[0])

        # result = np.array(data, dtype=object)
        # expected = np.array(list(data), dtype=object)
        # tm.assert_numpy_array_equal(result, expected)

    def test_copy(self, data):
        # GH#27083 removing deep keyword from EA.copy
        assert np.any(data[0] != data[1])
        result = data.copy()

        data[1] = data[0]
        assert np.any(result[1] != result[0])

    def test_view(self, data):
        # view with no dtype should return a shallow copy, *not* the same
        #  object
        assert np.any(data[1] != data[0])

        result = data.view()
        assert result is not data
        assert type(result) == type(data)

        result[1] = result[0]
        assert np.all(data[1] == data[0])

    #     # check specifically that the `dtype` kwarg is accepted
    #     data.view(dtype=None)


class TestConstructors(base.BaseConstructorsTests):
    pass


@pytest.mark.skip("WIP")
class TestReshaping(base.BaseReshapingTests):
    def test_concat_mixed_dtypes(self, data):
        # https://github.com/pandas-dev/pandas/issues/20762
        df1 = pd.DataFrame({"A": data[:3]})
        df2 = pd.DataFrame({"A": [1, 2, 3]})
        df3 = pd.DataFrame({"A": ["a", "b", "c"]}).astype("category")
        dfs = [df1, df2, df3]

        # dataframes - this will not work !!!
        # result = pd.concat(dfs)
        # expected = pd.concat([x.astype(object) for x in dfs])
        # self.assert_frame_equal(result, expected)

        # series
        result = pd.concat([x["A"] for x in dfs])
        expected = pd.concat([x["A"].astype(object) for x in dfs])
        self.assert_series_equal(result, expected)

        # simple test for just EA and one other
        result = pd.concat([df1, df2])
        expected = pd.concat([df1.astype("object"), df2.astype("object")])
        self.assert_frame_equal(result, expected)

        result = pd.concat([df1["A"], df2["A"]])
        expected = pd.concat([df1["A"].astype("object"), df2["A"].astype("object")])
        self.assert_series_equal(result, expected)

    @pytest.mark.skip("Not relevant")
    def test_merge_on_extension_array():
        pass

    @pytest.mark.skip("Not relevant")
    def test_merge_on_extension_array_duplicates():
        pass


class TestGetitem(base.BaseGetitemTests):
    def test_get(self, data):
        # GH 20882
        s = pd.Series(data, index=[2 * i for i in range(len(data))])
        assert np.all(s.get(4) == s.iloc[2])

    def test_take_sequence(self, data):
        result = pd.Series(data)[[0, 1, 3]]
        assert np.all(result.iloc[0] == data[0])

    def test_take(self, data, na_value, na_cmp):
        result = data.take([0, -1])
        assert result.dtype == data.dtype
        assert np.all(result[0] == data[0])

    def test_item(self, data):
        # https://github.com/pandas-dev/pandas/pull/30175
        s = pd.Series(data)
        result = s[:1].item()
        assert np.all(result == data[0])


class TestSetitem(base.base.BaseExtensionTests):  # (base.BaseSetitemTests):
    def test_setitem_scalar_series(self, data, box_in_series):
        if box_in_series:
            data = pd.Series(data)
        data[0] = data[1]
        assert np.all(data[0] == data[1])

    def test_setitem_sequence(self, data, box_in_series):
        if box_in_series:
            data = pd.Series(data)
        original = data.copy()

        data[[0, 1]] = [data[1], data[0]]
        assert np.all(data[0] == original[1])

    def test_setitem_empty_indxer(self, data, box_in_series):
        if box_in_series:
            data = pd.Series(data)
        original = data.copy()
        data[np.array([], dtype=int)] = data[:0]  # []
        self.assert_equal(data, original)

    def test_setitem_sequence_broadcasts(self, data, box_in_series):
        if box_in_series:
            data = pd.Series(data)
        data[[0, 1]] = data[2]
        assert np.all(data[0] == data[2])

    @pytest.mark.parametrize("setter", ["loc", "iloc"])
    def test_setitem_scalar(self, data, setter):
        arr = pd.Series(data)
        setter = getattr(arr, setter)
        operator.setitem(setter, 0, data[1])
        assert np.all(arr[0] == data[1])


# class TestMissing(base.BaseMissingTests):
#     pass


# class TestMissing(base.BaseCastingTests):
#     pass


# class TestMissing(base.BaseGroupbyTests):
#     pass


# class TestMissing(base.BaseParsingTests):
#     pass


# class TestMissing(base.BaseMethodsTests):
#     pass


# class TestMissing(base.BaseArithmeticOpsTests):
#     pass


# class TestMissing(base.BaseComparisonOpsTests):
#     pass


# class TestMissing(base.BaseOpsUtil):
#     pass


# class TestMissing(base.BaseUnaryOpsTests):
#     pass


# class TestMissing(base.BasePrintingTests):
#     pass


# class TestMissing(base.BaseBooleanReduceTests):
#     pass


# class TestMissing(base.BaseNoReduceTests):
#     pass


# class TestMissing(base.BaseNumericReduceTests):
#     pass
