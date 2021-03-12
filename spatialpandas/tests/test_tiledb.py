import pandas as pd
from hypothesis import HealthCheck, given, settings

from spatialpandas import GeoDataFrame
from spatialpandas.io import read_tiledb, to_tiledb

from .geometry.strategies import st_geodataframe

hyp_settings = settings(
    deadline=None,
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)


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
