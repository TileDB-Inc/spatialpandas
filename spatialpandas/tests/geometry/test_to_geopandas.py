import pytest
import geopandas as gp
from hypothesis import given
from pandas.testing import assert_series_equal

from .strategies import (
    hyp_settings,
    st_line_array,
    st_multiline_array,
    st_multipoint_array,
    st_multipolygon_array,
    st_point_array,
    st_polygon_array,
    st_ring_array,
)
from spatialpandas import GeoSeries


@given(st_point_array(astype=gp.GeoSeries))
@hyp_settings
def test_point_array_to_geopandas(gp_point):
    result = GeoSeries(gp_point, dtype='point').to_geopandas()
    assert_series_equal(result, gp_point)


@given(st_multipoint_array(astype=gp.GeoSeries))
@hyp_settings
def test_multipoint_array_to_geopandas(gp_multipoint):
    result = GeoSeries(gp_multipoint, dtype='multipoint').to_geopandas()
    assert_series_equal(result, gp_multipoint)


@given(st_line_array(astype=gp.GeoSeries))
@hyp_settings
def test_line_array_to_geopandas(gp_line):
    result = GeoSeries(gp_line, dtype='line').to_geopandas()
    assert_series_equal(result, gp_line)


@given(st_ring_array(astype=gp.GeoSeries))
@hyp_settings
def test_ring_array_to_geopandas(gp_ring):
    result = GeoSeries(gp_ring, dtype='ring').to_geopandas()
    assert_series_equal(result, gp_ring)


@given(st_multiline_array(astype=gp.GeoSeries))
@hyp_settings
def test_multiline_array_to_geopandas(gp_multiline):
    result = GeoSeries(gp_multiline, dtype='multiline').to_geopandas()
    assert_series_equal(result, gp_multiline)


@pytest.mark.slow
@given(st_polygon_array(astype=gp.GeoSeries))
@hyp_settings
def test_polygon_array_to_geopandas(gp_polygon):
    result = GeoSeries(gp_polygon, dtype='polygon').to_geopandas()
    assert_series_equal(result, gp_polygon)


@pytest.mark.slow
@given(st_multipolygon_array(astype=gp.GeoSeries))
@hyp_settings
def test_multipolygon_array_to_geopandas(gp_multipolygon):
    result = GeoSeries(gp_multipolygon, dtype='multipolygon').to_geopandas()
    assert_series_equal(result, gp_multipolygon)
