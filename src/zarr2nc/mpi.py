from __future__ import annotations

import argparse
import textwrap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zarr2nc-mpi",
        description="Placeholder for a future MPI-enabled single-file Zarr to netCDF4 writer.",
    )
    parser.add_argument("source", nargs="?", help="Input Zarr store path")
    parser.add_argument("target", nargs="?", help="Output .nc path")
    parser.add_argument("--dim", default="time", help="Primary partition dimension")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    parser.parse_args(argv)
    message = """
    zarr2nc-mpi is intentionally a scaffold in this skeleton.

    Intended implementation:
      1. Require a conda/mamba environment with MPI-enabled HDF5, netCDF-C,
         netCDF4-python, and mpi4py.
      2. Use xarray/zarr for metadata and source chunk reads.
      3. Use netCDF4.Dataset(..., parallel=True, comm=...) for the output file.
      4. Create dimensions, variables, and attributes collectively on all ranks.
      5. Assign non-overlapping hyperslabs to ranks and write in collective mode.

    The non-MPI path is zarr2nc-shards + zarr2nc-concat.
    """
    raise SystemExit(textwrap.dedent(message).strip())


if __name__ == "__main__":
    main()
