from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from zarr2nc.encoding import parse_dim_int_map


def expand_parts(parts: list[str]) -> list[str]:
    expanded: list[str] = []
    for item in parts:
        matches = sorted(glob.glob(item))
        expanded.extend(matches or [item])
    return expanded


def require_executable(executable: str, *, purpose: str) -> None:
    if shutil.which(executable) is None:
        raise FileNotFoundError(
            f"{purpose} requires {executable!r}, but it is not on PATH. "
            "Use the portable Python backend, run inside the pixi environment, "
            "or load the appropriate HPC module."
        )


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def shards_from_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest = load_manifest(path)
    return sorted(
        manifest.get("shards", []),
        key=lambda shard: (shard.get("start", 0), shard.get("stop", 0), shard.get("path", "")),
    )


def parts_from_manifest(path: str | Path) -> list[str]:
    manifest_path = Path(path)
    base_dir = manifest_path.parent
    parts: list[str] = []
    for shard in shards_from_manifest(manifest_path):
        raw_path = Path(shard["path"])
        parts.append(str(raw_path if raw_path.is_absolute() else base_dir / raw_path))
    return parts


def build_ncrcat_command(
    parts: list[str],
    output: str,
    *,
    ncrcat: str = "ncrcat",
    overwrite: bool = False,
    netcdf4: bool = True,
) -> list[str]:
    if not parts:
        raise ValueError("At least one part file is required")
    cmd = [ncrcat]
    if overwrite:
        cmd.append("-O")
    if netcdf4:
        cmd.append("-4")
    cmd.extend(parts)
    cmd.append(output)
    return cmd


def build_nccopy_command(
    input_path: str,
    output_path: str,
    *,
    nccopy: str = "nccopy",
    chunks: dict[str, int] | None = None,
    deflate: int | None = None,
    shuffle: bool = False,
    in_memory: bool = False,
    kind: str = "nc4",
) -> list[str]:
    cmd = [nccopy, "-k", kind]
    if deflate is not None:
        cmd.extend(["-d", str(deflate)])
    if shuffle:
        cmd.append("-s")
    if chunks:
        chunk_text = ",".join(f"{dim}/{size}" for dim, size in chunks.items())
        cmd.extend(["-c", chunk_text])
    if in_memory:
        cmd.append("-w")
    cmd.extend([input_path, output_path])
    return cmd


def concat_parts(
    parts: list[str],
    output: str,
    *,
    overwrite: bool = False,
    ncrcat: str = "ncrcat",
    netcdf4: bool = True,
) -> str:
    output_path = Path(output)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite")
    require_executable(ncrcat, purpose="The ncrcat merge backend")
    cmd = build_ncrcat_command(parts, output, ncrcat=ncrcat, overwrite=overwrite, netcdf4=netcdf4)
    subprocess.run(cmd, check=True)
    return output


def repack_file(
    input_path: str,
    output_path: str,
    *,
    overwrite: bool = False,
    nccopy: str = "nccopy",
    chunks: dict[str, int] | None = None,
    deflate: int | None = None,
    shuffle: bool = False,
    in_memory: bool = False,
    kind: str = "nc4",
) -> str:
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite")
    require_executable(nccopy, purpose="nccopy repacking")
    cmd = build_nccopy_command(
        input_path,
        output_path,
        nccopy=nccopy,
        chunks=chunks,
        deflate=deflate,
        shuffle=shuffle,
        in_memory=in_memory,
        kind=kind,
    )
    subprocess.run(cmd, check=True)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zarr2nc-concat",
        description=(
            "Concatenate shard .nc files with NCO ncrcat and optionally repack with nccopy."
        ),
    )
    parser.add_argument(
        "parts",
        nargs="*",
        help="Part files or glob patterns. Quote globs if needed.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Read part files from zarr2nc-shards.json and sort by shard offsets.",
    )
    parser.add_argument("--output", required=True, help="Merged output .nc file")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ncrcat", default="ncrcat", help="Path to ncrcat executable")
    parser.add_argument("--no-netcdf4", action="store_true", help="Do not pass -4 to ncrcat")
    parser.add_argument(
        "--repack-output",
        default=None,
        help="Optional final output path after nccopy repacking. Leaves --output as intermediate.",
    )
    parser.add_argument("--nccopy", default="nccopy", help="Path to nccopy executable")
    parser.add_argument("--nccopy-kind", default="nc4", help='nccopy -k value, default "nc4"')
    parser.add_argument("--nccopy-deflate", type=int, default=None, help="nccopy -d value")
    parser.add_argument("--nccopy-shuffle", action="store_true", help="Pass -s to nccopy")
    parser.add_argument(
        "--nccopy-chunks",
        default=None,
        help='nccopy chunk map, for example "time=24,lat=200,lon=200"',
    )
    parser.add_argument("--nccopy-in-memory", action="store_true", help="Pass -w to nccopy")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    parts = parts_from_manifest(args.manifest) if args.manifest else expand_parts(args.parts)
    if not parts:
        parser.error("provide part files or --manifest")
    concat_parts(
        parts,
        args.output,
        overwrite=args.overwrite,
        ncrcat=args.ncrcat,
        netcdf4=not args.no_netcdf4,
    )

    if args.repack_output:
        repack_file(
            args.output,
            args.repack_output,
            overwrite=args.overwrite,
            nccopy=args.nccopy,
            chunks=parse_dim_int_map(args.nccopy_chunks),
            deflate=args.nccopy_deflate,
            shuffle=args.nccopy_shuffle,
            in_memory=args.nccopy_in_memory,
            kind=args.nccopy_kind,
        )


if __name__ == "__main__":
    main()
