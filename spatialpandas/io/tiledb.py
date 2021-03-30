__all__ = ["to_tiledb", "read_tiledb", "to_tiledb_cloud"]

import json
from collections import defaultdict
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import tiledb

from ..geodataframe import GeoDataFrame
from ..geometry import GeometryDtype
from ..geometry.flattened import FlatGeometryArray, FlatGeometryDtype


def to_tiledb(df: GeoDataFrame, uri: str, ctx: Optional[tiledb.Ctx] = None) -> None:
    df = _flatten_geometry_columns(df)
    varlen_types = frozenset(_iter_flat_geometry_dtypes(df))
    return tiledb.from_pandas(uri, df, varlen_types=varlen_types, ctx=ctx)


def read_tiledb(
    uri: str,
    columns: Optional[Sequence[str]] = None,
    ctx: Optional[tiledb.Ctx] = None,
) -> GeoDataFrame:
    df = tiledb.open_dataframe(uri, attrs=columns, use_arrow=False, ctx=ctx)
    new_columns = {}
    for name, column in df.items():
        if isinstance(column.dtype, FlatGeometryDtype):
            new_columns[name] = column.astype(column.dtype.geometry_dtype)
        else:
            new_columns[name] = column

    return GeoDataFrame(new_columns, index=df.index)


def to_tiledb_cloud(
    df: GeoDataFrame, uri: str, npartitions: int = 8, ctx: Optional[tiledb.Ctx] = None
) -> None:
    npartitions = min(len(df), npartitions)

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
        dfs=map(_flatten_geometry_columns, partition_dfs),
        # dense array if the df index is range(0, len(df))
        sparse=not df.index.equals(pd.RangeIndex(len(df))),
        varlen_types=frozenset(_iter_flat_geometry_dtypes(df)),
        ctx=ctx,
    )

    # save the partition ranges and geometry bounds as metadata
    with tiledb.open(uri, mode="w", ctx=ctx) as a:
        a.meta["spatialpandas"] = json.dumps(
            {
                "partition_ranges": partition_ranges,
                "partition_bounds": partition_bounds,
            }
        )


def _iter_flat_geometry_dtypes(df: GeoDataFrame) -> Iterable[FlatGeometryDtype]:
    for name, dtype in df.dtypes.items():
        if isinstance(dtype, FlatGeometryDtype):
            yield dtype
        elif isinstance(dtype, GeometryDtype):
            yield FlatGeometryDtype(dtype)


def _flatten_geometry_columns(df: GeoDataFrame) -> GeoDataFrame:
    new_columns = {
        name: FlatGeometryArray(column.values)
        for name, column in df.items()
        if isinstance(column.dtype, GeometryDtype)
    }
    return df.assign(**new_columns) if new_columns else df


def _from_multiple_pandas(uri: str, dfs: Iterable[pd.DataFrame], **kwargs) -> None:
    # TODO: parallelize this loop
    mode = "ingest"
    for df in dfs:
        # row_start_idx is used only for dense arrays, it's ignored for sparse
        tiledb.from_pandas(
            uri, df, mode=mode, full_domain=True, row_start_idx=df.index[0], **kwargs
        )
        mode = "append"


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
