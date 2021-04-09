__all__ = ["to_tiledb", "read_tiledb"]

import itertools as it
import json
from collections import defaultdict
from numbers import Real
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tiledb

try:
    from tiledb.cloud.compute import Delayed
except ImportError:  # pragma: no cover
    Delayed = None

from ..geodataframe import GeoDataFrame
from ..geometry import GeometryDtype
from ..geometry.flattened import FlatGeometryArray, FlatGeometryDtype


def to_tiledb(
    df: GeoDataFrame,
    uri: str,
    npartitions: int = 1,
    ctx: Optional[tiledb.Ctx] = None,
    tiledb_cloud_kwargs: Optional[Mapping[str, Any]] = None,
) -> None:
    from_pandas_kwargs = dict(
        # write a dense array only if the df index is range(0, len(df))
        sparse=not df.index.equals(pd.RangeIndex(len(df))),
        varlen_types=frozenset(
            FlatGeometryDtype(dtype)
            for dtype in df.dtypes
            if isinstance(dtype, GeometryDtype)
        ),
        ctx=ctx,
    )

    npartitions = min(len(df), npartitions)
    if npartitions <= 1:
        return tiledb.from_pandas(
            uri, _flatten_geometry_columns(df), **from_pandas_kwargs
        )

    # sort the dataframe by the first index level if not already sorted
    # TODO: figure out how to take into account all levels for MultiIndex
    sorted_index, indices = df.index.sortlevel(level=0)
    if not sorted_index.equals(df.index):
        df = df.iloc[indices]

    # partition the dataframe into sub-dataframes based on the first index level
    index_values = sorted_index.get_level_values(level=0).values
    partition_slices = iter_partition_slices(index_values, npartitions)

    # for each partition:
    # - record its range
    # - select the respective sub-dataframe
    # - compute and record the bounds of each geometry column
    partition_dfs = []
    partition_ranges = []
    all_bounds = defaultdict(list)
    for partition_slice in partition_slices:
        partition_df = df.loc[partition_slice]
        partition_dfs.append(partition_df)
        partition_ranges.append((partition_slice.start, partition_slice.stop))
        for name, column in partition_df.items():
            if isinstance(column.dtype, GeometryDtype):
                all_bounds[name].append(column.total_bounds)

    # stack the bounds of all partitions for each geometry column
    partition_bounds = {
        name: {"data": bounds_list, "columns": ("x0", "y0", "x1", "y1")}
        for name, bounds_list in all_bounds.items()
    }

    # write all partitions to tiledb
    _from_multiple_pandas(
        uri,
        map(_flatten_geometry_columns, partition_dfs),
        from_pandas_kwargs,
        tiledb_cloud_kwargs,
    )

    # save the partition ranges and geometry bounds as metadata
    with tiledb.open(uri, mode="w", ctx=ctx) as a:
        a.meta["spatialpandas"] = json.dumps(
            {
                "partition_ranges": partition_ranges,
                "partition_bounds": partition_bounds,
            }
        )


def read_tiledb(
    uri: str,
    *,
    columns: Optional[Sequence[str]] = None,
    geometry: Optional[str] = None,
    bounds: Optional[Tuple[Real, Real, Real, Real]] = None,
    ctx: Optional[tiledb.Ctx] = None,
) -> GeoDataFrame:
    kwargs = dict(
        geometry=geometry,
        open_df_kwargs=dict(attrs=columns, ctx=ctx),
    )

    # load the metadata for partitions and geometry column bounds
    partition_slices, partition_bounds = load_partition_metadata(uri)

    # no partition metadata found: read the whole geodataframe
    if not partition_slices:
        return _read_geodataframe(uri, slice(None, None), **kwargs)

    if bounds is not None:
        if geometry is None:
            # get an empty geodataframe to determine the geometry column
            geometry = _read_geodataframe(uri, **kwargs).geometry.name

        # get the bounds of partitions for the geometry column
        partition_bounds_df = partition_bounds.get(geometry)
        if partition_bounds_df is not None:
            # unpack bounds coordinates and make sure x0 < x1 & y0 < y1
            x0, y0, x1, y1 = bounds
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            # determine which partitions have non-zero overlap with the bounds rectangle
            inds = ~(
                (partition_bounds_df.x1 < x0)
                | (partition_bounds_df.y1 < y0)
                | (partition_bounds_df.x0 > x1)
                | (partition_bounds_df.y0 > y1)
            )
            partition_slices = np.array(partition_slices)[inds].tolist()

    return _read_geodataframe(uri, *partition_slices, **kwargs)


# ========= helper functions ====================================================


def _read_geodataframe(
    uri: str,
    *partition_slices: slice,
    geometry: Optional[str] = None,
    open_df_kwargs: Optional[Mapping[str, Any]] = None,
) -> GeoDataFrame:
    named_columns = {}
    idx = list(partition_slices) if partition_slices else None
    df = tiledb.open_dataframe(uri, idx=idx, use_arrow=False, **(open_df_kwargs or {}))
    for name, column in df.items():
        # unflatten the geometry columns
        if isinstance(column.dtype, FlatGeometryDtype):
            column = column.astype(column.dtype.geometry_dtype)
        named_columns[name] = column
    return GeoDataFrame(named_columns, geometry=geometry)


def _flatten_geometry_columns(df: GeoDataFrame) -> GeoDataFrame:
    new_columns = {
        name: FlatGeometryArray(column.values)
        for name, column in df.items()
        if isinstance(column.dtype, GeometryDtype)
    }
    return df.assign(**new_columns) if new_columns else df


def _from_multiple_pandas(
    uri: str,
    dfs: Iterable[pd.DataFrame],
    from_pandas_kwargs: Optional[Mapping[str, Any]] = None,
    tiledb_cloud_kwargs: Optional[Mapping[str, Any]] = None,
) -> None:
    def from_pandas(df: pd.DataFrame, mode) -> None:
        tiledb.from_pandas(
            uri,
            df,
            mode=mode,
            full_domain=True,
            # row_start_idx is used only for dense arrays, it's ignored for sparse
            row_start_idx=df.index[0],
            **(from_pandas_kwargs or {}),
        )

    iter_dfs = iter(dfs)

    if Delayed is None or tiledb_cloud_kwargs is None:
        from_pandas(next(iter_dfs), mode="ingest")
        for df in iter_dfs:
            from_pandas(df, mode="append")
    else:
        ingest_task = Delayed(from_pandas, **tiledb_cloud_kwargs)(
            next(iter_dfs), mode="ingest"
        )
        tasks = [ingest_task]
        for df in iter_dfs:
            append_task = Delayed(from_pandas, **tiledb_cloud_kwargs)(df, mode="append")
            append_task.depends_on(ingest_task)
            tasks.append(append_task)
        ingest_task.compute()


def iter_partition_slices(array: np.ndarray, n: int) -> Iterable[slice]:
    """Split `array` values into (at most) `n` partitions of approximately equal sum.

    Each partition is represented as `slice(start, stop)` where *both `start` and `stop`
    are inclusive*.

    Note: the resulting partitioning is not guaranteed to be optimal.

    :param array: The values to split
    :param n: Max number of partitions
    :return: An iterable of `array` value slices, one slice per partition
    """
    unique, counts = np.unique(array, return_counts=True)
    # convert unique elements from numpy to pure python instances
    unique = unique.tolist()

    num_unique = len(unique)
    if num_unique <= n:
        # if equal or more partitions than unique elements, each partition
        # consists of a single unique element
        return (slice(item, item) for item in unique)

    # Adapted from https://stackoverflow.com/a/54024280/240525
    # finds the indices where the cumulative sums are sandwiched
    p_size = len(array) / n
    boundaries = np.searchsorted(
        counts.cumsum(), np.arange(1, n) * p_size, side="right"
    )

    start_indices = np.r_[0, boundaries]
    # if there are duplicates in start_indices, increment them
    for i, start_index in enumerate(start_indices[:-1]):
        if start_indices[i + 1] <= start_index:
            start_indices[i + 1] = start_index + 1

    # inclusive stop_indices: 1 less from the next start index
    stop_indices = np.r_[start_indices[1:], num_unique]
    stop_indices -= 1

    # sanity checks
    assert np.all(start_indices <= stop_indices)
    for indices in start_indices, stop_indices:
        assert np.all(indices >= 0)
        assert np.all(indices < num_unique)
        assert np.all(np.diff(indices) > 0)

    return (
        slice(unique[start], unique[stop])
        for start, stop in zip(start_indices, stop_indices)
    )


def load_partition_metadata(
    uri: str, ctx: Optional[tiledb.Ctx] = None
) -> Tuple[Sequence[slice], Mapping[str, pd.DataFrame]]:
    with tiledb.open(uri, ctx=ctx) as a:
        try:
            spatialpandas = json.loads(a.meta["spatialpandas"])
        except KeyError:
            return (), {}

    partition_slices = tuple(it.starmap(slice, spatialpandas["partition_ranges"]))

    partition_bounds = spatialpandas["partition_bounds"]
    for name, bounds_dict in partition_bounds.items():
        partition_bounds[name] = pd.DataFrame(**bounds_dict)

    return partition_slices, partition_bounds
