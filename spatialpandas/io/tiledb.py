from typing import Optional, Sequence

import tiledb

from ..geodataframe import GeoDataFrame
from ..geometry import GeometryDtype
from ..geometry.flattened import FlatGeometryArray, FlatGeometryDtype


def to_tiledb(df: GeoDataFrame, uri: str, **kwargs) -> None:
    new_columns = {}
    varlen_types = set(kwargs.pop("varlen_types", ()))
    for name, column in df.iteritems():
        if isinstance(column.dtype, GeometryDtype):
            new_column = FlatGeometryArray(column.values)
            new_columns[name] = new_column
            varlen_types.add(new_column.dtype)

    if new_columns:
        df = df.assign(**new_columns)

    return tiledb.from_pandas(uri, df, varlen_types=varlen_types, **kwargs)


def read_tiledb(
    uri: str,
    columns: Optional[Sequence[str]] = None,
    ctx: Optional[tiledb.Ctx] = None,
) -> GeoDataFrame:
    df = tiledb.open_dataframe(uri, attrs=columns, use_arrow=False, ctx=ctx)
    new_columns = {}
    for name, column in df.iteritems():
        if isinstance(column.dtype, FlatGeometryDtype):
            new_columns[name] = column.astype(column.dtype.geometry_dtype)
        else:
            new_columns[name] = column

    return GeoDataFrame(new_columns, index=df.index)
