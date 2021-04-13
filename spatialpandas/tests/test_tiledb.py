import string

import dask.dataframe as dd
import numpy as np
import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from .geometry.strategies import st_bounds, st_geodataframe
from spatialpandas import GeoDataFrame
from spatialpandas.io.tiledb import (
    Delayed,
    iter_partition_slices,
    load_partition_metadata,
    read_tiledb,
    to_tiledb,
)

hyp_settings = settings(
    deadline=None,
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
)


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


def test_iter_partition_slices_for_unique_values():
    a = np.array(list(string.ascii_lowercase))

    assert list(iter_partition_slices(a, 1)) == [slice("a", "z")]
    assert list(iter_partition_slices(a, 2)) == [slice("a", "m"), slice("n", "z")]
    assert list(iter_partition_slices(a, 3)) == [
        slice("a", "h"),
        slice("i", "q"),
        slice("r", "z"),
    ]
    assert list(iter_partition_slices(a, 4)) == [
        slice("a", "f"),
        slice("g", "m"),
        slice("n", "s"),
        slice("t", "z"),
    ]
    assert list(iter_partition_slices(a, 5)) == [
        slice("a", "e"),
        slice("f", "j"),
        slice("k", "o"),
        slice("p", "t"),
        slice("u", "z"),
    ]
    assert list(iter_partition_slices(a, 6)) == [
        slice("a", "d"),
        slice("e", "h"),
        slice("i", "m"),
        slice("n", "q"),
        slice("r", "u"),
        slice("v", "z"),
    ]
    assert list(iter_partition_slices(a, 7)) == [
        slice("a", "c"),
        slice("d", "g"),
        slice("h", "k"),
        slice("l", "n"),
        slice("o", "r"),
        slice("s", "v"),
        slice("w", "z"),
    ]
    assert list(iter_partition_slices(a, 8)) == [
        slice("a", "c"),
        slice("d", "f"),
        slice("g", "i"),
        slice("j", "m"),
        slice("n", "p"),
        slice("q", "s"),
        slice("t", "v"),
        slice("w", "z"),
    ]
    assert list(iter_partition_slices(a, 9)) == [
        slice("a", "b"),
        slice("c", "e"),
        slice("f", "h"),
        slice("i", "k"),
        slice("l", "n"),
        slice("o", "q"),
        slice("r", "t"),
        slice("u", "w"),
        slice("x", "z"),
    ]
    assert list(iter_partition_slices(a, 10)) == [
        slice("a", "b"),
        slice("c", "e"),
        slice("f", "g"),
        slice("h", "j"),
        slice("k", "m"),
        slice("n", "o"),
        slice("p", "r"),
        slice("s", "t"),
        slice("u", "w"),
        slice("x", "z"),
    ]


@given(df=st_geodataframe(min_size=8, max_size=20), npartitions=st.sampled_from([3, 7]))
@hyp_settings
def test_load_partition_metadata(df, npartitions, tmp_path_factory):
    with tmp_path_factory.mktemp("spatialpandas", numbered=True) as tmp_path:
        uri = str(tmp_path / "df.tdb")
        to_tiledb(df, uri, npartitions=npartitions)

        partition_ranges, partition_bounds = load_partition_metadata(uri)

        assert len(partition_ranges) == npartitions
        assert all(isinstance(r, slice) for r in partition_ranges)

        assert set(partition_bounds.keys()).issubset(df.columns)
        for partition_bounds_df in partition_bounds.values():
            assert tuple(partition_bounds_df.columns) == ("x0", "y0", "x1", "y1")
            assert len(partition_bounds_df) == npartitions


@given(
    df=st_geodataframe(min_size=8, max_size=20),
    pack=st.booleans(),
    npartitions=st.sampled_from([1, 3]),
    tiledb_cloud_kwargs=(
        st.sampled_from([None, {"local": True}]) if Delayed is not None else st.none()
    ),
)
@hyp_settings
def test_to_tiledb_read_tiledb_roundtrip(
    df, pack, npartitions, tiledb_cloud_kwargs, tmp_path_factory
):
    if pack:
        df.pack(inplace=True)

    with tmp_path_factory.mktemp("spatialpandas", numbered=True) as tmp_path:
        uri = str(tmp_path / "df.tdb")
        to_tiledb(
            df, uri, npartitions=npartitions, tiledb_cloud_kwargs=tiledb_cloud_kwargs
        )

        df_read = read_tiledb(uri)
        assert isinstance(df_read, GeoDataFrame)
        pd.testing.assert_frame_equal(df.sort_index(), df_read.sort_index())

        columns = ["a", "multilines", "polygons"]
        df_read = read_tiledb(uri, columns=columns)
        assert isinstance(df_read, GeoDataFrame)
        pd.testing.assert_frame_equal(df[columns].sort_index(), df_read.sort_index())


@given(
    df=st_geodataframe(min_size=8, max_size=20),
    geometry=st.sampled_from([None, "lines", "polygons"]),
    bounds=st_bounds(),
)
@hyp_settings
def test_read_tiledb_bounds(df, geometry, bounds, tmp_path_factory):
    df.set_geometry("polygons", inplace=True)
    df.pack(inplace=True)
    # if there are any Hilbert distance ties there is no unique ordering, so skip the test
    assume(len(set(df.index)) == len(df.index))

    with tmp_path_factory.mktemp("spatialpandas", numbered=True) as tmp_path:
        uri = str(tmp_path / "df.tdb")
        to_tiledb(df, uri, npartitions=3)

        df_read = read_tiledb(uri, geometry=geometry, bounds=bounds)
        assert isinstance(df_read, GeoDataFrame)
        expected_geometry = geometry or "points"
        assert df_read.geometry.name == expected_geometry

        # create a DaskGeoDataFrame with the same partitions created by to_tiledb
        p_slices, p_bounds = load_partition_metadata(uri)
        divisions = [*(s.start for s in p_slices), p_slices[-1].stop]
        ddf = (
            dd.from_pandas(df, npartitions=1)
            .repartition(divisions=divisions)
            .set_geometry(expected_geometry)
        )

        # and use the `ddf.cx_partitions` indexer to get the expected bounded partitions
        x0, y0, x1, y1 = bounds
        expected_df = ddf.cx_partitions[x0:x1, y0:y1].compute()

        # verify they are equal
        pd.testing.assert_frame_equal(expected_df.sort_index(), df_read.sort_index())
