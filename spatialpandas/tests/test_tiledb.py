import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings, strategies as st

from spatialpandas import GeoDataFrame
from spatialpandas.io import read_tiledb, read_tiledb_cloud, to_tiledb, to_tiledb_cloud
from spatialpandas.io.tiledb import iter_partition_slices, load_partition_metadata

from .geometry.strategies import st_geodataframe

hyp_settings = settings(
    deadline=None,
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
)


def pack_geodataframe(df, *, geometry=None, p=15, inplace=False):
    if geometry is None:
        geometry = df.geometry
    hilbert_distance = geometry.hilbert_distance(p=p)
    df2 = df.set_index(hilbert_distance, inplace=inplace)
    (df2 if df2 is not None else df).rename_axis("hilbert_distance", inplace=True)
    return df2


def test_iter_partition_slices():
    f = np.full  # f(frequency, value)
    a = np.r_[f(5, 1), f(3, 4), f(4, 6), f(2, 8), f(1, 10), f(2, 12), f(3, 15)]
    np.random.shuffle(a)

    assert list(iter_partition_slices(a, 1)) == [slice(1, 15)]
    assert list(iter_partition_slices(a, 2)) == [slice(1, 4), slice(6, 15)]
    assert list(iter_partition_slices(a, 3)) == [slice(1, 1), slice(4, 6), slice(8, 15)]
    assert list(iter_partition_slices(a, 4)) == [
        slice(1, 1),
        slice(4, 4),
        slice(6, 10),
        slice(12, 15),
    ]
    assert list(iter_partition_slices(a, 5)) == [
        slice(1, 1),
        slice(4, 4),
        slice(6, 6),
        slice(8, 10),
        slice(12, 15),
    ]
    assert list(iter_partition_slices(a, 6)) == [
        slice(1, 1),
        slice(4, 4),
        slice(6, 6),
        slice(8, 8),
        slice(10, 10),
        slice(12, 15),
    ]
    for p in range(7, len(a)):
        assert list(iter_partition_slices(a, p)) == [slice(i, i) for i in np.unique(a)]


@given(df=st_geodataframe())
@hyp_settings
def test_tiledb(df, tmp_path_factory):
    df.index.name = "range_idx"
    with tmp_path_factory.mktemp("spatialpandas", numbered=True) as tmp_path:
        uri = str(tmp_path / "df.tdb")
        to_tiledb(df, uri)

        df_read = read_tiledb(uri)
        assert isinstance(df_read, GeoDataFrame)
        pd.testing.assert_frame_equal(df, df_read)

        columns = ["a", "multilines", "polygons"]
        df_read = read_tiledb(uri, columns=columns)
        assert isinstance(df_read, GeoDataFrame)
        pd.testing.assert_frame_equal(df[columns], df_read)


@given(df=st_geodataframe(min_size=8, max_size=20), pack=st.booleans())
@hyp_settings
def test_to_tiledb_cloud(df, pack, tmp_path_factory):
    if pack:
        pack_geodataframe(df, inplace=True)
    with tmp_path_factory.mktemp("spatialpandas", numbered=True) as tmp_path:
        uri = str(tmp_path / "df.tdb")
        npartitions = 3
        to_tiledb_cloud(df, uri, npartitions=npartitions)

        df_read = read_tiledb(uri)
        assert isinstance(df_read, GeoDataFrame)
        # the dataframe rows are generally reordered
        pd.testing.assert_frame_equal(df.sort_index(), df_read.sort_index())

        columns = ["a", "multilines", "polygons"]
        df_read = read_tiledb(uri, columns=columns)
        assert isinstance(df_read, GeoDataFrame)
        # the dataframe rows are generally reordered
        pd.testing.assert_frame_equal(df[columns].sort_index(), df_read.sort_index())


@given(df=st_geodataframe(min_size=8, max_size=20), pack=st.booleans())
@hyp_settings
def test_read_tiledb_cloud(df, pack, tmp_path_factory):
    if pack:
        pack_geodataframe(df, inplace=True)
    with tmp_path_factory.mktemp("spatialpandas", numbered=True) as tmp_path:
        uri = str(tmp_path / "df.tdb")
        npartitions = 3
        to_tiledb_cloud(df, uri, npartitions=npartitions)

        df_read = read_tiledb_cloud(uri)
        assert isinstance(df_read, GeoDataFrame)
        # the dataframe rows are generally reordered
        pd.testing.assert_frame_equal(df.sort_index(), df_read.sort_index())

        columns = ["a", "multilines", "polygons"]
        df_read = read_tiledb_cloud(uri, columns=columns)
        assert isinstance(df_read, GeoDataFrame)
        # the dataframe rows are generally reordered
        pd.testing.assert_frame_equal(df[columns].sort_index(), df_read.sort_index())


@given(df=st_geodataframe(min_size=3))
@hyp_settings
def test_load_partition_metadata(df, tmp_path_factory):
    with tmp_path_factory.mktemp("spatialpandas", numbered=True) as tmp_path:
        uri = str(tmp_path / "df.tdb")
        npartitions = 3
        to_tiledb_cloud(df, uri, npartitions=npartitions)

        partition_ranges, partition_bounds = load_partition_metadata(uri)

        assert len(partition_ranges) == npartitions
        assert all(isinstance(r, slice) for r in partition_ranges)

        assert set(partition_bounds.keys()).issubset(df.columns)
        for partition_bounds_df in partition_bounds.values():
            assert tuple(partition_bounds_df.columns) == ("x0", "y0", "x1", "y1")
            assert len(partition_bounds_df) == npartitions
