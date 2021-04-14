import dask
import dask.dataframe as dd
import hypothesis.strategies as hs
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, Phase, Verbosity, given, settings

from spatialpandas import GeoDataFrame
from spatialpandas.dask import DaskGeoDataFrame
from spatialpandas.io import read_parquet, read_parquet_dask, to_parquet

from .geometry.strategies import st_bounds, st_geodataframe


dask.config.set(scheduler="single-threaded")

hyp_settings = settings(
    deadline=None,
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)


@given(df=st_geodataframe())
@hyp_settings
def test_parquet(df, tmp_path_factory):
    df.index.name = "range_idx"
    with tmp_path_factory.mktemp("spatialpandas", numbered=True) as tmp_path:
        path = tmp_path / "df.parq"
        to_parquet(df, path)

        df_read = read_parquet(path)
        assert isinstance(df_read, GeoDataFrame)
        pd.testing.assert_frame_equal(df, df_read)

        columns = ["a", "multilines", "polygons"]
        df_read = read_parquet(str(path), columns=columns)
        assert isinstance(df_read, GeoDataFrame)
        pd.testing.assert_frame_equal(df[columns], df_read)


@given(df=st_geodataframe(column_names=("points", "lines", "a")))
@hyp_settings
def test_parquet_dask(df, tmp_path_factory):
    with tmp_path_factory.mktemp("spatialpandas", numbered=True) as tmp_path:
        ddf = dd.from_pandas(df, npartitions=3)

        path = tmp_path / "ddf.parq"
        ddf.to_parquet(str(path))
        ddf_read = read_parquet_dask(str(path))

        assert isinstance(ddf_read, DaskGeoDataFrame)
        assert ddf_read.geometry.name == "points"

        # Check that partition bounds were loaded
        assert set(ddf_read._partition_bounds) == {"points", "lines"}
        nonempty = np.nonzero(np.asarray(ddf.map_partitions(len).compute() > 0))[0]
        for column in "points", "lines":
            expected_partition_bounds = (
                ddf[column].partition_bounds.iloc[nonempty].reset_index(drop=True)
            )
            expected_partition_bounds.index.name = "partition"
            pd.testing.assert_frame_equal(
                expected_partition_bounds,
                ddf_read._partition_bounds[column],
            )


@given(
    st_geodataframe(
        min_size=10, max_size=40, column_names=("multipoints", "multilines", "a")
    )
)
@settings(deadline=None, max_examples=30)
def test_pack_partitions(df):
    df = df.set_geometry("multilines")
    ddf = dd.from_pandas(df, npartitions=3)

    # Pack partitions
    ddf_packed = ddf.pack_partitions(npartitions=4)

    # Check the number of partitions
    assert ddf_packed.npartitions == 4

    # Check that rows are now sorted in order of hilbert distance
    total_bounds = df.multilines.total_bounds
    hilbert_distances = (
        ddf_packed.multilines.map_partitions(
            lambda s: s.hilbert_distance(total_bounds=total_bounds)
        )
        .compute()
        .values
    )

    # Compute expected total_bounds
    expected_distances = np.sort(
        df.multilines.hilbert_distance(total_bounds=total_bounds).values
    )

    np.testing.assert_equal(expected_distances, hilbert_distances)


@pytest.mark.slow
@given(
    df=st_geodataframe(
        min_size=60, max_size=100, column_names=("multipoints", "multilines", "a")
    ),
    use_temp_format=hs.booleans(),
)
@settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow],
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target],
    verbosity=Verbosity.verbose,
)
def test_pack_partitions_to_parquet(df, use_temp_format, tmp_path_factory):
    with tmp_path_factory.mktemp("spatialpandas", numbered=True) as tmp_path:
        df = df.set_geometry("multilines")
        ddf = dd.from_pandas(df, npartitions=3)

        path = tmp_path / "ddf.parq"
        if use_temp_format:
            (tmp_path / "scratch").mkdir(parents=True, exist_ok=True)
            tempdir_format = str(tmp_path / "scratch" / "part-{uuid}-{partition:03d}")
        else:
            tempdir_format = None

        _retry_args = dict(
            wait_exponential_multiplier=10,
            wait_exponential_max=20000,
            stop_max_attempt_number=4,
        )

        ddf_packed = ddf.pack_partitions_to_parquet(
            str(path),
            npartitions=12,
            tempdir_format=tempdir_format,
            _retry_args=_retry_args,
        )

        # Check the number of partitions (< 4 can happen in the case of empty partitions)
        assert ddf_packed.npartitions <= 12

        # Check that rows are now sorted in order of hilbert distance
        total_bounds = df.multilines.total_bounds
        hilbert_distances = (
            ddf_packed.multilines.map_partitions(
                lambda s: s.hilbert_distance(total_bounds=total_bounds)
            )
            .compute()
            .values
        )

        # Compute expected total_bounds
        expected_distances = np.sort(
            df.multilines.hilbert_distance(total_bounds=total_bounds).values
        )

        np.testing.assert_equal(expected_distances, hilbert_distances)
        assert ddf_packed.geometry.name == "multipoints"

        # Read columns
        columns = ["a", "multilines"]
        ddf_read_cols = read_parquet_dask(path, columns=columns)
        pd.testing.assert_frame_equal(
            ddf_read_cols.compute(), ddf_packed[columns].compute()
        )


@pytest.mark.slow
@given(
    df1=st_geodataframe(
        min_size=10, max_size=40, column_names=("multipoints", "multilines", "a")
    ),
    df2=st_geodataframe(
        min_size=10, max_size=40, column_names=("multipoints", "multilines", "a")
    ),
)
@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_pack_partitions_to_parquet_glob(df1, df2, tmp_path_factory):
    with tmp_path_factory.mktemp("spatialpandas", numbered=True) as tmp_path:
        df1 = df1.set_geometry("multilines")
        ddf1 = dd.from_pandas(df1, npartitions=3)
        path1 = tmp_path / "ddf1.parq"
        ddf_packed1 = ddf1.pack_partitions_to_parquet(str(path1), npartitions=3)

        df2 = df2.set_geometry("multilines")
        ddf2 = dd.from_pandas(df2, npartitions=3)
        path2 = tmp_path / "ddf2.parq"
        ddf_packed2 = ddf2.pack_partitions_to_parquet(str(path2), npartitions=4)

        # Load both packed datasets with glob
        ddf_globbed = read_parquet_dask(tmp_path / "ddf*.parq", geometry="multilines")

        # Check the number of partitions (< 7 can happen in the case of empty partitions)
        assert ddf_globbed.npartitions <= 7

        # Check contents
        expected_df = pd.concat([ddf_packed1.compute(), ddf_packed2.compute()])
        df_globbed = ddf_globbed.compute()
        pd.testing.assert_frame_equal(df_globbed, expected_df)

        # Check partition bounds
        expected_bounds = {
            "multipoints": pd.concat(
                [
                    ddf_packed1._partition_bounds["multipoints"],
                    ddf_packed2._partition_bounds["multipoints"],
                ]
            ).reset_index(drop=True),
            "multilines": pd.concat(
                [
                    ddf_packed1._partition_bounds["multilines"],
                    ddf_packed2._partition_bounds["multilines"],
                ]
            ).reset_index(drop=True),
        }
        expected_bounds["multipoints"].index.name = "partition"
        expected_bounds["multilines"].index.name = "partition"
        pd.testing.assert_frame_equal(
            expected_bounds["multipoints"], ddf_globbed._partition_bounds["multipoints"]
        )

        pd.testing.assert_frame_equal(
            expected_bounds["multilines"], ddf_globbed._partition_bounds["multilines"]
        )

        assert ddf_globbed.geometry.name == "multilines"


@pytest.mark.slow
@given(
    df1=st_geodataframe(
        min_size=10, max_size=40, column_names=("multipoints", "multilines", "a")
    ),
    df2=st_geodataframe(
        min_size=10, max_size=40, column_names=("multipoints", "multilines", "a")
    ),
    bounds=st_bounds(),
)
@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_pack_partitions_to_parquet_list_bounds(df1, df2, bounds, tmp_path_factory):
    with tmp_path_factory.mktemp("spatialpandas", numbered=True) as tmp_path:
        df1 = df1.set_geometry("multilines")
        ddf1 = dd.from_pandas(df1, npartitions=3)
        path1 = tmp_path / "ddf1.parq"
        ddf_packed1 = ddf1.pack_partitions_to_parquet(str(path1), npartitions=3)

        df2 = df2.set_geometry("multilines")
        ddf2 = dd.from_pandas(df2, npartitions=3)
        path2 = tmp_path / "ddf2.parq"
        ddf_packed2 = ddf2.pack_partitions_to_parquet(str(path2), npartitions=4)

        # Load both packed datasets with glob
        ddf_read = read_parquet_dask(
            [str(tmp_path / "ddf1.parq"), str(tmp_path / "ddf2.parq")],
            geometry="multipoints",
            bounds=bounds,
        )

        # Check the number of partitions (< 7 can happen in the case of empty partitions)
        assert ddf_read.npartitions <= 7

        # Check contents
        xslice = slice(bounds[0], bounds[2])
        yslice = slice(bounds[1], bounds[3])
        expected_df = pd.concat(
            [
                ddf_packed1.cx_partitions[xslice, yslice].compute(),
                ddf_packed2.cx_partitions[xslice, yslice].compute(),
            ]
        )
        df_read = ddf_read.compute()
        pd.testing.assert_frame_equal(df_read, expected_df)

        # Compute expected partition bounds
        points_bounds = pd.concat(
            [
                ddf_packed1._partition_bounds["multipoints"],
                ddf_packed2._partition_bounds["multipoints"],
            ]
        ).reset_index(drop=True)

        x0, y0, x1, y1 = bounds
        x0, x1 = (x0, x1) if x0 <= x1 else (x1, x0)
        y0, y1 = (y0, y1) if y0 <= y1 else (y1, y0)
        partition_inds = ~(
            (points_bounds.x1 < x0)
            | (points_bounds.y1 < y0)
            | (points_bounds.x0 > x1)
            | (points_bounds.y0 > y1)
        )
        points_bounds = points_bounds[partition_inds].reset_index(drop=True)

        lines_bounds = (
            pd.concat(
                [
                    ddf_packed1._partition_bounds["multilines"],
                    ddf_packed2._partition_bounds["multilines"],
                ]
            )
            .reset_index(drop=True)[partition_inds]
            .reset_index(drop=True)
        )
        points_bounds.index.name = "partition"
        lines_bounds.index.name = "partition"

        # Check partition bounds
        pd.testing.assert_frame_equal(
            points_bounds, ddf_read._partition_bounds["multipoints"]
        )

        pd.testing.assert_frame_equal(
            lines_bounds, ddf_read._partition_bounds["multilines"]
        )

        # Check active geometry column
        assert ddf_read.geometry.name == "multipoints"
