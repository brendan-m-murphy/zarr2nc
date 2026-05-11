from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import xarray as xr

from zarr2nc import __version__
from zarr2nc.config import NetcdfFormat, SliceOptions, ZarrFormat
from zarr2nc.encoding import (
    parse_consolidated,
    parse_csv,
    parse_dim_int_map,
    parse_open_chunks,
    parse_zarr_format,
)
from zarr2nc.slice_writer import convert_slice


def plan_shards(
    size: int,
    *,
    shard_size: int | None = None,
    num_shards: int | None = None,
) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` pairs covering ``range(size)`` exactly once."""

    if size < 0:
        raise ValueError("size must be non-negative")
    if shard_size is not None and shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if num_shards is not None and num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_size is not None and num_shards is not None:
        raise ValueError("Specify only one of shard_size or num_shards")
    if size == 0:
        return []

    if shard_size is None and num_shards is None:
        shard_size = size

    if shard_size is not None:
        return [(start, min(start + shard_size, size)) for start in range(0, size, shard_size)]

    assert num_shards is not None
    # Even split without numpy as a hard dependency for this helper.
    starts = [(i * size) // num_shards for i in range(num_shards)]
    stops = [((i + 1) * size) // num_shards for i in range(num_shards)]
    return [(start, stop) for start, stop in zip(starts, stops, strict=True) if start < stop]


def source_dim_size(
    source: str,
    *,
    dim: str,
    group: str | None,
    consolidated: bool | None,
    zarr_format: ZarrFormat | None = None,
) -> int:
    ds = xr.open_zarr(
        source,
        group=group,
        consolidated=consolidated,
        zarr_format=zarr_format,
        chunks=None,
        decode_cf=False,
        mask_and_scale=False,
        decode_times=False,
    )
    try:
        if dim not in ds.sizes:
            raise ValueError(f"Dimension {dim!r} not found in source dataset")
        return int(ds.sizes[dim])
    finally:
        ds.close()


def build_slice_options(
    *,
    source: str,
    output_dir: Path,
    dim: str,
    shards: list[tuple[int, int]],
    prefix: str,
    suffix: str,
    group: str | None,
    consolidated: bool | None,
    zarr_format: ZarrFormat | None,
    open_chunks: object,
    output_chunks: dict[str, int] | None,
    format: str,
    compression: str | None,
    complevel: int,
    shuffle: bool,
    float_dtype: str | None,
    unlimited_dims: tuple[str, ...],
    drop_variables: tuple[str, ...],
    decode_cf: bool,
    scheduler: str,
    overwrite: bool,
) -> list[SliceOptions]:
    width = max(6, len(str(len(shards))))
    options: list[SliceOptions] = []
    for index, (start, stop) in enumerate(shards):
        target = output_dir / f"{prefix}{index:0{width}d}{suffix}"
        options.append(
            SliceOptions(
                source=source,
                target=str(target),
                dim=dim,
                start=start,
                stop=stop,
                group=group,
                consolidated=consolidated,
                zarr_format=zarr_format,
                open_chunks=open_chunks,
                output_chunks=output_chunks,
                format=cast(NetcdfFormat, format),
                compression=compression,
                complevel=complevel,
                shuffle=shuffle,
                float_dtype=float_dtype,
                unlimited_dims=unlimited_dims,
                drop_variables=drop_variables,
                decode_cf=decode_cf,
                scheduler=scheduler,
                overwrite=overwrite,
            )
        )
    return options


def write_manifest(output_dir: Path, options: list[SliceOptions]) -> Path:
    return write_shards_manifest(output_dir, options)


def write_shards_manifest(
    output_dir: Path,
    options: list[SliceOptions],
    *,
    source: str | None = None,
    group: str | None = None,
    consolidated: bool | None = None,
    split_dim: str | None = None,
    encoding: dict[str, Any] | None = None,
) -> Path:
    output_dir = output_dir.resolve()
    manifest = {
        "version": 1,
        "tool": "zarr2nc",
        "tool_version": __version__,
        "source": source or (options[0].source if options else None),
        "group": group,
        "consolidated": consolidated,
        "zarr_format": options[0].zarr_format if options else None,
        "split_dim": split_dim or (options[0].dim if options else None),
        "encoding": encoding or {},
        "shards": [
            {
                "path": os.path.relpath(Path(opt.target).resolve(), output_dir),
                "dim": opt.dim,
                "start": opt.start,
                "stop": opt.stop,
            }
            for opt in options
        ],
    }
    path = output_dir / "zarr2nc-shards.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def _convert_slice_from_dict(payload: dict) -> str:
    return convert_slice(SliceOptions(**payload))


def convert_shards(options: list[SliceOptions], *, jobs: int) -> list[str]:
    if jobs <= 0:
        raise ValueError("jobs must be positive")

    if jobs == 1:
        return [convert_slice(option) for option in options]

    results: list[str] = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(_convert_slice_from_dict, asdict(option)) for option in options]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zarr2nc-shards",
        description=(
            "Split an xarray-created Zarr store along one dimension and write many netCDF "
            "files in parallel, one writer process per output file."
        ),
    )
    parser.add_argument("source", help="Input Zarr store path")
    parser.add_argument("output_dir", help="Directory for part_*.nc files")
    parser.add_argument("--dim", required=True, help="Dimension to split, usually time")
    split = parser.add_mutually_exclusive_group()
    split.add_argument("--shard-size", type=int, default=None)
    split.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1, help="Number of writer processes")
    parser.add_argument("--prefix", default="part_")
    parser.add_argument("--suffix", default=".nc")
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
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create the output directory and manifest but do not write part files.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    consolidated = parse_consolidated(args.consolidated)
    zarr_format = parse_zarr_format(args.zarr_format)
    source_size = source_dim_size(
        args.source,
        dim=args.dim,
        group=args.group,
        consolidated=consolidated,
        zarr_format=zarr_format,
    )
    shards = plan_shards(source_size, shard_size=args.shard_size, num_shards=args.num_shards)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compression = None if args.compression == "none" else args.compression
    scheduler = "single-threaded" if args.scheduler == "synchronous" else args.scheduler
    options = build_slice_options(
        source=args.source,
        output_dir=output_dir,
        dim=args.dim,
        shards=shards,
        prefix=args.prefix,
        suffix=args.suffix,
        group=args.group,
        consolidated=consolidated,
        zarr_format=zarr_format,
        open_chunks=parse_open_chunks(args.open_chunks),
        output_chunks=parse_dim_int_map(args.chunks),
        format=args.format,
        compression=compression,
        complevel=args.complevel,
        shuffle=not args.no_shuffle,
        float_dtype=args.float_dtype,
        unlimited_dims=parse_csv(args.unlimited_dim),
        drop_variables=parse_csv(args.drop_variable),
        decode_cf=args.decode_cf,
        scheduler=scheduler,
        overwrite=args.overwrite,
    )

    if not args.no_manifest:
        write_shards_manifest(
            output_dir,
            options,
            source=args.source,
            group=args.group,
            consolidated=consolidated,
            split_dim=args.dim,
            encoding={
                "chunks": parse_dim_int_map(args.chunks),
                "compression": compression,
                "complevel": args.complevel,
                "shuffle": not args.no_shuffle,
                "float_dtype": args.float_dtype,
                "format": args.format,
            },
        )

    if not args.dry_run:
        convert_shards(options, jobs=args.jobs)


if __name__ == "__main__":
    main()
