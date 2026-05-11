from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import cast

from zarr2nc.concat import parts_from_manifest, require_executable
from zarr2nc.config import NetcdfFormat
from zarr2nc.doctor import collect_diagnostics, format_diagnostics
from zarr2nc.encoding import (
    parse_consolidated,
    parse_csv,
    parse_dim_int_map,
    parse_open_chunks,
    parse_zarr_format,
)
from zarr2nc.merge import MergeBackend, merge_parts
from zarr2nc.shards import (
    build_slice_options,
    convert_shards,
    plan_shards,
    source_dim_size,
    write_shards_manifest,
)
from zarr2nc.validation import (
    ValidationMode,
    format_report,
    raise_if_invalid,
    validate_output,
    validate_parts,
)


def _add_conversion_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dim", required=True, help="Dimension to split, usually time")
    split = parser.add_mutually_exclusive_group()
    split.add_argument("--shard-size", type=int, default=None)
    split.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1, help="Number of writer processes")
    parser.add_argument("--backend", choices=["python", "ncrcat", "none"], default="python")
    parser.add_argument("--work-dir", default=None, help="Directory for intermediate part files")
    parser.add_argument("--keep-parts", action="store_true", help="Keep intermediate part files")
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
    parser.add_argument("--validate", choices=["none", "quick", "full"], default="none")
    parser.add_argument("--ncrcat", default="ncrcat", help="Path to ncrcat executable")
    parser.add_argument("--overwrite", action="store_true")


def _parts_dir(
    target: Path,
    *,
    work_dir: str | None,
    keep_parts: bool,
    backend: MergeBackend,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if work_dir:
        return Path(work_dir), None
    if keep_parts or backend == "none":
        return target.parent / f"{target.stem}_parts", None
    temp = tempfile.TemporaryDirectory(prefix=f".{target.stem}.parts.", dir=target.parent)
    return Path(temp.name), temp


def run_convert(args: argparse.Namespace) -> None:
    backend = cast(MergeBackend, args.backend)
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if backend == "ncrcat":
        require_executable(args.ncrcat, purpose="The ncrcat merge backend")

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    parts_dir, temp_dir = _parts_dir(
        target,
        work_dir=args.work_dir,
        keep_parts=args.keep_parts,
        backend=backend,
    )
    parts_dir.mkdir(parents=True, exist_ok=True)

    consolidated = parse_consolidated(args.consolidated)
    zarr_format = parse_zarr_format(args.zarr_format)
    source_size = source_dim_size(
        args.source,
        dim=args.dim,
        group=args.group,
        consolidated=consolidated,
        zarr_format=zarr_format,
    )
    num_shards = args.num_shards
    if args.shard_size is None and num_shards is None and args.jobs > 1:
        num_shards = args.jobs
    shards = plan_shards(source_size, shard_size=args.shard_size, num_shards=num_shards)

    compression = None if args.compression == "none" else args.compression
    output_chunks = parse_dim_int_map(args.chunks)
    unlimited_dims = parse_csv(args.unlimited_dim)
    scheduler = "single-threaded" if args.scheduler == "synchronous" else args.scheduler

    options = build_slice_options(
        source=args.source,
        output_dir=parts_dir,
        dim=args.dim,
        shards=shards,
        prefix=args.prefix,
        suffix=args.suffix,
        group=args.group,
        consolidated=consolidated,
        zarr_format=zarr_format,
        open_chunks=parse_open_chunks(args.open_chunks),
        output_chunks=output_chunks,
        format=args.format,
        compression=compression,
        complevel=args.complevel,
        shuffle=not args.no_shuffle,
        float_dtype=args.float_dtype,
        unlimited_dims=unlimited_dims,
        drop_variables=parse_csv(args.drop_variable),
        decode_cf=args.decode_cf,
        scheduler=scheduler,
        overwrite=args.overwrite,
    )
    manifest_path = write_shards_manifest(
        parts_dir,
        options,
        source=args.source,
        group=args.group,
        consolidated=consolidated,
        split_dim=args.dim,
        encoding={
            "chunks": output_chunks,
            "compression": compression,
            "complevel": args.complevel,
            "shuffle": not args.no_shuffle,
            "float_dtype": args.float_dtype,
            "format": args.format,
        },
    )

    try:
        convert_shards(options, jobs=args.jobs)
        part_paths = parts_from_manifest(manifest_path)
        merge_parts(
            part_paths,
            target,
            backend=backend,
            dim=args.dim,
            overwrite=args.overwrite,
            output_chunks=output_chunks,
            format=cast(NetcdfFormat, args.format),
            compression=compression,
            complevel=args.complevel,
            shuffle=not args.no_shuffle,
            float_dtype=args.float_dtype,
            unlimited_dims=unlimited_dims,
            ncrcat=args.ncrcat,
        )

        if args.validate != "none":
            mode = cast(ValidationMode, args.validate)
            if backend == "none":
                report = validate_parts(
                    args.source,
                    manifest_path,
                    group=args.group,
                    consolidated=consolidated,
                    zarr_format=zarr_format,
                    decode_cf=args.decode_cf,
                    mode=mode,
                )
            else:
                report = validate_output(
                    args.source,
                    target,
                    dim=args.dim,
                    group=args.group,
                    consolidated=consolidated,
                    zarr_format=zarr_format,
                    decode_cf=args.decode_cf,
                    mode=mode,
                )
            print(format_report(report))
            raise_if_invalid(report)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
        elif not args.keep_parts and backend != "none" and args.work_dir is None:
            shutil.rmtree(parts_dir, ignore_errors=True)


def run_validate(args: argparse.Namespace) -> None:
    consolidated = parse_consolidated(args.consolidated)
    zarr_format = parse_zarr_format(args.zarr_format)
    mode = cast(ValidationMode, args.mode)
    if args.manifest:
        report = validate_parts(
            args.source,
            args.manifest,
            group=args.group,
            consolidated=consolidated,
            zarr_format=zarr_format,
            decode_cf=args.decode_cf,
            mode=mode,
        )
    else:
        if not args.output:
            raise ValueError("validate requires OUTPUT unless --manifest is provided")
        report = validate_output(
            args.source,
            args.output,
            dim=args.dim,
            group=args.group,
            consolidated=consolidated,
            zarr_format=zarr_format,
            decode_cf=args.decode_cf,
            mode=mode,
        )
    print(format_report(report))
    raise_if_invalid(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zarr2nc",
        description="Convert xarray-created Zarr stores to netCDF4/HDF5 products.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    convert = subcommands.add_parser("convert", help="Convert a Zarr store to .nc")
    convert.add_argument("source", help="Input Zarr store path")
    convert.add_argument("output", help="Output .nc path, or nominal target when --backend none")
    _add_conversion_options(convert)
    convert.set_defaults(func=run_convert)

    validate = subcommands.add_parser("validate", help="Validate a converted output")
    validate.add_argument("source", help="Input Zarr store path")
    validate.add_argument("output", nargs="?", help="Output .nc path")
    validate.add_argument("--manifest", default=None, help="Validate part files from a manifest")
    validate.add_argument("--dim", required=True, help="Primary split dimension")
    validate.add_argument("--group", default=None, help="Optional Zarr group")
    validate.add_argument(
        "--zarr-format",
        default="auto",
        choices=["auto", "2", "3"],
        help="Input Zarr format. Defaults to auto-detect.",
    )
    validate.add_argument(
        "--consolidated",
        default="auto",
        choices=["auto", "true", "false"],
        help="Whether to use consolidated Zarr metadata",
    )
    validate.add_argument(
        "--decode-cf",
        action="store_true",
        help="Decode CF metadata while validating. Default is false.",
    )
    validate.add_argument("--mode", choices=["quick", "full"], default="quick")
    validate.set_defaults(func=run_validate)

    doctor = subcommands.add_parser("doctor", help="Report package and backend availability")
    doctor.set_defaults(func=lambda _args: print(format_diagnostics(collect_diagnostics())))

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
