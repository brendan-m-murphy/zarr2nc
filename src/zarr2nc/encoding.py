from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any

import xarray as xr

from zarr2nc.config import ZarrFormat

_MISSING = object()


def parse_csv(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(item.strip() for item in text.split(",") if item.strip())


def parse_dim_int_map(text: str | None) -> dict[str, int] | None:
    """Parse CLI text such as ``time=24,lat=200,lon=200``."""

    if not text:
        return None

    out: dict[str, int] = {}
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected DIM=SIZE item, got {item!r}")
        dim, raw_value = item.split("=", 1)
        dim = dim.strip()
        if not dim:
            raise ValueError(f"Expected non-empty dimension name in {item!r}")
        value = int(raw_value)
        if value <= 0:
            raise ValueError(f"Chunk size for {dim!r} must be positive, got {value}")
        out[dim] = value
    return out


def parse_consolidated(text: str) -> bool | None:
    mapping = {
        "auto": None,
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "1": True,
        "0": False,
    }
    try:
        return mapping[text.lower()]
    except KeyError as exc:
        raise ValueError("consolidated must be one of: auto, true, false") from exc


def parse_zarr_format(text: str) -> ZarrFormat | None:
    mapping: dict[str, ZarrFormat | None] = {"auto": None, "2": 2, "3": 3}
    try:
        return mapping[text.lower()]
    except KeyError as exc:
        raise ValueError("zarr-format must be one of: auto, 2, 3") from exc


def parse_open_chunks(text: str) -> Any:
    """Parse the xarray.open_zarr chunks option from CLI text.

    ``auto`` leaves Dask chunking up to xarray/Dask, ``native`` uses Zarr chunks,
    and ``none`` opens eager/non-Dask arrays. A DIM=SIZE map is also accepted.
    """

    lowered = text.lower()
    if lowered == "auto":
        return "auto"
    if lowered == "native":
        return {}
    if lowered == "none":
        return None
    return parse_dim_int_map(text)


def clear_xarray_encodings(ds: xr.Dataset) -> xr.Dataset:
    """Drop source-store encodings before writing to netCDF/HDF5.

    Zarr encodings contain fields that do not map cleanly to netCDF. The output
    encoding is rebuilt explicitly by :func:`make_encoding`.
    """

    try:
        return ds.drop_encoding()
    except AttributeError:
        ds = ds.copy(deep=False)
        ds.encoding.clear()
        for variable in ds.variables.values():
            variable.encoding.clear()
        return ds


def extract_fill_values(ds: xr.Dataset) -> tuple[xr.Dataset, dict[str, Any]]:
    """Move ``_FillValue`` attrs into an encoding map.

    Xarray treats ``_FillValue`` as an encoding concept when writing netCDF.
    When opening with ``decode_cf=False`` it may appear in ``attrs`` instead.
    """

    ds = ds.copy(deep=False)
    fill_values: dict[str, Any] = {}

    for raw_name in ds.variables:
        name = str(raw_name)
        attrs = dict(ds[raw_name].attrs)
        fill_value = attrs.pop("_FillValue", _MISSING)
        if fill_value is not _MISSING:
            fill_values[name] = fill_value
            ds[raw_name].attrs = attrs

    return ds, fill_values


def make_encoding(
    ds: xr.Dataset,
    *,
    chunks: dict[str, int] | None,
    compression: str | None,
    complevel: int,
    shuffle: bool,
    float_dtype: str | None,
    fill_values: dict[str, Any] | None = None,
    compress_coordinates: bool = False,
) -> dict[str, dict[str, Any]]:
    """Build an xarray ``to_netcdf`` encoding dictionary.

    This intentionally implements a conservative policy. It is better for this
    tool to produce boring netCDF files that collaborators can open than to rely
    on exotic HDF5 filters or exact Zarr codec transfer.
    """

    fill_values = fill_values or {}
    encoding: dict[str, dict[str, Any]] = {}

    for raw_name, da in ds.variables.items():
        name = str(raw_name)
        enc: dict[str, Any] = {}

        if name in fill_values:
            enc["_FillValue"] = fill_values[name]
        elif raw_name not in ds.data_vars:
            enc["_FillValue"] = None

        if chunks and da.ndim:
            enc["chunksizes"] = tuple(
                min(chunks.get(str(dim), da.sizes[dim]), da.sizes[dim]) for dim in da.dims
            )

        should_compress = (
            compression is not None
            and da.ndim > 0
            and da.dtype.kind in "biufc"
            and (compress_coordinates or name in ds.data_vars)
        )
        if should_compress:
            if compression == "gzip":
                enc.update({"zlib": True, "complevel": complevel, "shuffle": shuffle})
            elif compression == "lzf":
                enc.update({"compression": "lzf"})
            else:
                enc.update({"compression": compression, "compression_opts": complevel})

        if float_dtype and da.dtype.kind == "f":
            enc["dtype"] = float_dtype

        if enc:
            encoding[name] = enc

    return encoding


def ensure_valid_slice(start: int, stop: int, size: int, dim: str) -> None:
    if start < 0:
        raise ValueError(f"start must be >= 0 for {dim!r}; got {start}")
    if stop < start:
        raise ValueError(f"stop must be >= start for {dim!r}; got start={start}, stop={stop}")
    if stop > size:
        raise ValueError(f"stop={stop} exceeds size {size} for dimension {dim!r}")


def normalize_unlimited_dims(
    unlimited_dims: Iterable[Hashable],
    default_dim: Hashable,
) -> tuple[Hashable, ...]:
    dims = tuple(dict.fromkeys(unlimited_dims))
    return dims or (default_dim,)
