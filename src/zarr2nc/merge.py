from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import xarray as xr

from zarr2nc.concat import concat_parts
from zarr2nc.config import NetcdfFormat
from zarr2nc.encoding import (
    clear_xarray_encodings,
    extract_fill_values,
    make_encoding,
    normalize_unlimited_dims,
)

MergeBackend = Literal["python", "ncrcat", "none"]


class MergeAlignmentError(ValueError):
    """Raised when Python merge cannot align shard dimensions."""


def _temporary_target(target: Path) -> Path:
    return target.with_name(f".{target.name}.tmp.{os.getpid()}")


def _is_varying_dimension_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "cannot reindex or align along dimension" in message
        and "conflicting dimension sizes" in message
    )


def _raise_helpful_alignment_error(exc: Exception) -> None:
    raise MergeAlignmentError(
        "Cannot merge parts because a non-concatenation dimension has different sizes "
        "across files. If that dimension is identified by a label variable, such as "
        "sitenames(nsite), merge with xarray by promoting the label variable to an index "
        "before concatenation, then reset the index before writing NetCDF."
    ) from exc


def merge_python_parts(
    parts: Sequence[str],
    output: str | Path,
    *,
    dim: str,
    overwrite: bool = False,
    output_chunks: dict[str, int] | None = None,
    format: NetcdfFormat = "NETCDF4",
    compression: str | None = "gzip",
    complevel: int = 2,
    shuffle: bool = True,
    float_dtype: str | None = None,
    unlimited_dims: Sequence[str] = (),
    atomic_write: bool = True,
) -> str:
    """Merge shard files with only Python dependencies."""

    if not parts:
        raise ValueError("At least one part file is required")

    output_path = Path(output)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_target = _temporary_target(output_path) if atomic_write else output_path
    if write_target.exists():
        write_target.unlink()

    ds: xr.Dataset | None = None
    try:
        try:
            ds = xr.open_mfdataset(
                list(parts),
                engine="h5netcdf",
                combine="nested",
                concat_dim=dim,
                chunks=None,
                decode_cf=False,
                mask_and_scale=False,
                decode_times=False,
            )
        except Exception as exc:
            if _is_varying_dimension_error(exc):
                _raise_helpful_alignment_error(exc)
            raise

        ds = clear_xarray_encodings(ds)
        ds, fill_values = extract_fill_values(ds)
        encoding = make_encoding(
            ds,
            chunks=output_chunks,
            compression=compression,
            complevel=complevel,
            shuffle=shuffle,
            float_dtype=float_dtype,
            fill_values=fill_values,
        )
        ds.to_netcdf(
            str(write_target),
            engine="h5netcdf",
            format=format,
            encoding=encoding,
            unlimited_dims=normalize_unlimited_dims(unlimited_dims, dim),
        )
        if atomic_write:
            os.replace(write_target, output_path)
    except Exception:
        if atomic_write and write_target.exists():
            write_target.unlink()
        raise
    finally:
        if ds is not None:
            ds.close()

    return str(output_path)


def merge_parts(
    parts: Sequence[str],
    output: str | Path,
    *,
    backend: MergeBackend,
    dim: str,
    overwrite: bool = False,
    output_chunks: dict[str, int] | None = None,
    format: NetcdfFormat = "NETCDF4",
    compression: str | None = "gzip",
    complevel: int = 2,
    shuffle: bool = True,
    float_dtype: str | None = None,
    unlimited_dims: Sequence[str] = (),
    ncrcat: str = "ncrcat",
) -> str | None:
    if backend == "none":
        return None
    if backend == "ncrcat":
        return concat_parts(list(parts), str(output), overwrite=overwrite, ncrcat=ncrcat)
    return merge_python_parts(
        parts,
        output,
        dim=dim,
        overwrite=overwrite,
        output_chunks=output_chunks,
        format=format,
        compression=compression,
        complevel=complevel,
        shuffle=shuffle,
        float_dtype=float_dtype,
        unlimited_dims=unlimited_dims,
    )
