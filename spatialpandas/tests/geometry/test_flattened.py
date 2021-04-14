import numpy as np
from hypothesis import given, settings, strategies

from spatialpandas import GeoSeries
from spatialpandas.geometry.flattened import (
    flatten_geometry_array,
    unflatten_geometry_array,
)

from .strategies import (
    st_line_array,
    st_multiline_array,
    st_multipoint_array,
    st_multipolygon_array,
    st_point_array,
    st_polygon_array,
    st_ring_array,
)


@given(
    gp_array=strategies.one_of(
        st_point_array(min_size=1),
        st_multipoint_array(min_size=1),
        st_line_array(min_size=1),
        st_ring_array(min_size=3),
        st_multiline_array(min_size=1),
        st_polygon_array(min_size=1),
        st_multipolygon_array(min_size=1),
    )
)
@settings(deadline=None, max_examples=100)
def test_flatten_unflatten(gp_array):
    sp_array = GeoSeries(gp_array).values

    flattened = flatten_geometry_array(sp_array)
    assert flattened.shape == sp_array.shape
    for flat_array in flattened:
        assert flat_array.ndim == 1
        assert flat_array.dtype == sp_array.dtype.subtype

    unflattened = unflatten_geometry_array(flattened, sp_array.dtype)
    assert sp_array.__class__ is unflattened.__class__
    assert sp_array.dtype == unflattened.dtype
    np.testing.assert_array_equal(sp_array, unflattened)
