import re

import numpy as np
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    no_default,
    register_extension_dtype,
)
from pandas.api.types import is_dtype_equal
from pandas.core.dtypes.base import registry

from ..geometry import GeometryArray, GeometryDtype
from ..geometry.basefixed import GeometryFixed, GeometryFixedArray
from ..geometry.baselist import GeometryList, GeometryListArray


@register_extension_dtype
class FlatGeometryDtype(ExtensionDtype):
    _metadata = ("geometry_dtype",)

    type = np.ndarray
    na_value = None

    def __init__(self, geometry_dtype):
        self.geometry_dtype = geometry_dtype

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.geometry_dtype})"

    @property
    def name(self):
        return f"flat[{self.geometry_dtype}]"

    @property
    def subtype(self):
        return self.geometry_dtype.subtype

    @classmethod
    def construct_array_type(cls):
        return FlatGeometryArray

    @classmethod
    def construct_from_string(cls, string):
        msg = f"Cannot construct a '{cls.__name__}' from '{string}'"

        match = re.match(fr"^flat\[(.+)\]$", string)
        if match is None:
            raise TypeError(msg)

        wrapped_dtype = registry.find(match.group(1))
        if wrapped_dtype is None:
            raise TypeError(msg)

        try:
            return cls(wrapped_dtype)
        except Exception:
            raise TypeError(msg)


class FlatGeometryArray(ExtensionArray):
    def __init__(self, geometry_array: GeometryArray):
        self._geometry_array = geometry_array
        self._dtype = FlatGeometryDtype(geometry_array.dtype)

    @property
    def dtype(self):
        return self._dtype

    def __len__(self):
        return len(self._geometry_array)

    def copy(self):
        return self.__class__(self._geometry_array)

    def isna(self):
        return self._geometry_array.isna()

    def to_numpy(self, dtype=None, copy=False, na_value=no_default):
        result = flatten_geometry_array(self._geometry_array, dtype)
        if na_value is not no_default:
            result[self.isna()] = na_value
        return result

    def astype(self, dtype, copy=True):
        if is_dtype_equal(dtype, self._dtype.geometry_dtype):
            return self._geometry_array
        return super().astype(dtype=dtype, copy=copy)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        if isinstance(scalars, cls):
            return scalars

        if isinstance(dtype, FlatGeometryDtype):
            dtype = dtype.geometry_dtype

        if isinstance(scalars, GeometryArray):
            geometry_array = scalars.astype(dtype, copy=False)
        elif isinstance(dtype, GeometryDtype):
            geometry_array = unflatten_geometry_array(scalars, dtype)
        else:
            raise TypeError(f"[Flat]GeometryDtype expected, {dtype} passed")

        return cls(geometry_array)


def flatten_geometry_array(geometry_array: GeometryArray, dtype=None) -> np.ndarray:
    if dtype is None:
        dtype = geometry_array.numpy_dtype

    if isinstance(geometry_array, GeometryFixedArray):
        flatten = _flatten_fixed_geometry
    elif isinstance(geometry_array, GeometryListArray):
        flatten = _flatten_list_geometry
    else:
        raise TypeError(f"Flattening '{geometry_array.__class__}' not supported")

    flat_arrays = np.empty(len(geometry_array), dtype="O")
    for i, geometry in enumerate(geometry_array):
        if geometry is not None:
            flat_arrays[i] = flatten(geometry).astype(dtype, copy=False)
    return flat_arrays


def _flatten_fixed_geometry(geometry: GeometryFixed) -> np.ndarray:
    return geometry.flat_values


def _flatten_list_geometry(geometry: GeometryList) -> np.ndarray:
    # For GeometryList subclasses with 0 nesting levels (MultiPoint, Line, Ring),
    # `.buffer_offsets` returns a length-1 tuple of offsets that include
    # all the flat values. These offsets are both unnecessary and mess up the
    # unflattening by creating an extra nesting level, so ignore them
    offsets = geometry.buffer_offsets if geometry._nesting_levels > 0 else ()
    parts = [(len(offsets), *map(len, offsets))]
    parts.extend(offsets)
    parts.append(geometry.buffer_values)
    return np.concatenate(parts)


def unflatten_geometry_array(
    ragged_array: np.ndarray, dtype: GeometryDtype
) -> GeometryArray:
    array_type = dtype.construct_array_type()

    if issubclass(array_type, GeometryFixedArray):
        return array_type(ragged_array)

    if issubclass(array_type, GeometryListArray):
        sub_arrays_list = []
        for flat_array in ragged_array:
            assert isinstance(flat_array, np.ndarray) and flat_array.ndim == 1
            offsets_list = []
            num_offsets = int(flat_array[0])
            flat_offset = 1 + num_offsets
            for offsets_length in flat_array[1:flat_offset]:
                next_flat_offset = flat_offset + int(offsets_length)
                offsets = flat_array[flat_offset:next_flat_offset].astype(np.int32)
                offsets_list.append(offsets)
                flat_offset = next_flat_offset

            sub_arrays = flat_array[flat_offset:]
            for offsets in reversed(offsets_list):
                sub_arrays = [sub_arrays[i:j] for i, j in zip(offsets, offsets[1:])]
            sub_arrays_list.append(sub_arrays)

        return array_type(sub_arrays_list)

    raise TypeError(f"Unflattening '{array_type}' not supported")
