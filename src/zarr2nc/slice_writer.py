from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import cast

import dask
import xarray as xr

from zarr2nc.config import NetcdfFormat, SliceOptions
from zarr2nc.encoding import (
    clear_xarray_encodings,
    ensure_valid_slice,
    extract_fill_values,
    make_encoding,
    normalize_unlimited_dims,
    parse_consolidated,
    parse_csv,
    parse_dim_int_map,
    parse_open_chunks,
    parse_zarr_format,
)


def open_source_dataset(options: SliceOptions) -> xr.Dataset:
    return xr.open_zarr(
        options.source,
        group=options.group,
        consolidated=options.consolidated,
        zarr_format=options.zarr_format,
        chunks=options.open_chunks,
        decode_cf=options.decode_cf,
        mask_and_scale=options.decode_cf,
        decode_times=options.decode_cf,
    )


def prepare_slice(ds: xr.Dataset, options: SliceOptions) -> xr.Dataset:
    if options.dim not in ds.sizes:
        raise ValueError(f"Dimension {options.dim!r} not found in source dataset")

    ensure_valid_slice(options.start, options.stop, ds.sizes[options.dim], options.dim)
    ds = ds.isel({options.dim: slice(options.start, options.stop)})

    if options.drop_variables:
        ds = ds.drop_vars(list(options.drop_variables), errors="ignore")

    return ds


def _temporary_target(target: Path) -> Path:
    return target.with_name(f".{target.name}.tmp.{os.getpid()}")


def convert_slice(options: SliceOptions) -> str:
    """Write one slice of an xarray-created Zarr store to one netCDF file."""

    target = options.target_path
    if target.exists() and not options.overwrite:
        raise FileExistsError(f"{target} already exists; pass overwrite=True or --overwrite")

    target.parent.mkdir(parents=True, exist_ok=True)

    ds = open_source_dataset(options)
    ds = prepare_slice(ds, options)
    ds = clear_xarray_encodings(ds)
    ds, fill_values = extract_fill_values(ds)

    encoding = make_encoding(
        ds,
        chunks=options.output_chunks,
        compression=options.compression,
        complevel=options.complevel,
        shuffle=options.shuffle,
        float_dtype=options.float_dtype,
        fill_values=fill_values,
    )
    unlimited_dims = normalize_unlimited_dims(options.unlimited_dims, options.dim)

    write_target = _temporary_target(target) if options.atomic_write else target
    if write_target.exists():
        write_target.unlink()

    try:
        with dask.config.set(scheduler=options.scheduler):
            ds.to_netcdf(
                str(write_target),
                engine="h5netcdf",
                format=options.format,
                encoding=encoding,
                unlimited_dims=unlimited_dims,
            )
        if options.atomic_write:
            os.replace(write_target, target)
    except Exception:
        if options.atomic_write and write_target.exists():
            write_target.unlink()
        raise

    return str(target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zarr2nc-slice",
        description="Write one slice of an xarray-created Zarr store to one netCDF4/HDF5 file.",
    )
    parser.add_argument("source", help="Input Zarr store path")
    parser.add_argument("target", help="Output .nc path")
    parser.add_argument("--dim", required=True, help="Dimension to slice, usually time")
    parser.add_argument("--start", required=True, type=int, help="Inclusive slice start index")
    parser.add_argument("--stop", required=True, type=int, help="Exclusive slice stop index")
    parser.add_argument("--group", default=None, help="Optional Zarr group")
    parser.add_argument(
        "--zarr-format",
        default="auto",
        choices=["auto", "2", "3"],
        help="Input Zarr format. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--consolidated",
        default="auto",
        choices=["auto", "true", "false"],
        help="Whether to use consolidated Zarr metadata",
    )
    parser.add_argument(
        "--open-chunks",
        default="auto",
        help='xarray.open_zarr chunks option: auto, native, none, or "dim=size,..."',
    )
    parser.add_argument(
        "--chunks",
        default=None,
        help='Output HDF5 chunk sizes, for example "time=24,lat=200,lon=200"',
    )
    parser.add_argument("--format", choices=["NETCDF4", "NETCDF4_CLASSIC"], default="NETCDF4")
    parser.add_argument(
        "--compression",
        default="gzip",
        help='Compression policy: "gzip", "lzf", another h5py filter name, or "none"',
    )
    parser.add_argument("--complevel", type=int, default=2)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--float-dtype", choices=["float32", "float64"], default=None)
    parser.add_argument(
        "--unlimited-dim",
        default=None,
        help='Comma-separated unlimited dimensions. Defaults to --dim. Example: "time"',
    )
    parser.add_argument("--drop-variable", default=None, help="Comma-separated variables to drop")
    parser.add_argument(
        "--decode-cf",
        action="store_true",
        help="Decode CF metadata on read. Default is false to preserve encoded values.",
    )
    parser.add_argument(
        "--scheduler",
        default="threads",
        choices=["threads", "single-threaded", "processes", "synchronous"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-atomic-write",
        action="store_true",
        help="Write directly to target instead of temp-file then rename.",
    )
    return parser


def options_from_args(args: argparse.Namespace) -> SliceOptions:
    compression = None if args.compression == "none" else args.compression
    return SliceOptions(
        source=args.source,
        target=args.target,
        dim=args.dim,
        start=args.start,
        stop=args.stop,
        group=args.group,
        consolidated=parse_consolidated(args.consolidated),
        zarr_format=parse_zarr_format(args.zarr_format),
        open_chunks=parse_open_chunks(args.open_chunks),
        output_chunks=parse_dim_int_map(args.chunks),
        format=cast(NetcdfFormat, args.format),
        compression=compression,
        complevel=args.complevel,
        shuffle=not args.no_shuffle,
        float_dtype=args.float_dtype,
        unlimited_dims=parse_csv(args.unlimited_dim),
        drop_variables=parse_csv(args.drop_variable),
        decode_cf=args.decode_cf,
        scheduler="single-threaded" if args.scheduler == "synchronous" else args.scheduler,
        overwrite=args.overwrite,
        atomic_write=not args.no_atomic_write,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    convert_slice(options_from_args(args))


if __name__ == "__main__":
    main()
