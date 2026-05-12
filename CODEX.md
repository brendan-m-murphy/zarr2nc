# CODEX.md

## Objective

This repository provides a standalone CLI for converting **xarray-flavour Zarr**
stores to collaborator-facing netCDF4/HDF5 products.

The v1 path must be portable with `uv` alone:

```text
xarray Dataset.to_zarr store
  -> zarr2nc convert writes independent shard files in parallel
  -> zarr2nc convert merges those shards with Python/xarray/h5netcdf
  -> final .nc file
```

External binaries are optional accelerators, not default requirements. In
particular, do not require CDO, NCO, MPI, `netCDF4`, preloaded HPC modules, or
MPI-enabled HDF5/netCDF-C for the normal conversion path.

## Repository Layout

```text
src/zarr2nc/cli.py            top-level zarr2nc convert/validate/doctor command
src/zarr2nc/encoding.py       CLI parsing helpers and netCDF encoding policy
src/zarr2nc/slice_writer.py   zarr2nc-slice implementation
src/zarr2nc/shards.py         zarr2nc-shards implementation and shard manifest
src/zarr2nc/merge.py          Python and ncrcat merge backends
src/zarr2nc/concat.py         ncrcat/nccopy command wrappers
src/zarr2nc/validation.py     source-vs-output validation helpers
src/zarr2nc/doctor.py         environment diagnostics
src/zarr2nc/mpi.py            placeholder for future true MPI single-file writer
tests/                        unit and lightweight integration tests
```

## User-Facing Commands

Portable default:

```bash
zarr2nc convert input.zarr output.nc \
  --dim time \
  --zarr-format auto \
  --jobs 8 \
  --chunks time=24,lat=200,lon=200 \
  --compression gzip \
  --complevel 1 \
  --validate quick \
  --overwrite
```

`--zarr-format auto` is the default and lets xarray detect the source store
format. Use `--zarr-format 2` or `--zarr-format 3` only when a workflow needs to
force the source store format.

Optional NCO backend when `ncrcat` is available:

```bash
zarr2nc convert input.zarr output.nc \
  --dim time \
  --jobs 8 \
  --backend ncrcat \
  --validate full \
  --overwrite
```

Part-file deliverable:

```bash
zarr2nc convert input.zarr output.nc \
  --dim time \
  --backend none \
  --shard-size 744 \
  --validate quick \
  --overwrite
```

Diagnostics and validation:

```bash
zarr2nc doctor
zarr2nc validate input.zarr output.nc --dim time --mode full
zarr2nc validate input.zarr --manifest output_parts/zarr2nc-shards.json --dim time
```

## Development Environment: uv Path

Use this path for normal coding and release checks.

```bash
uv venv
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run mypy src
```

The base package intentionally avoids `netCDF4` as a hard dependency. Use
`h5netcdf`/`h5py` for the normal writer and Python merge path.

## Development Environment: pixi Path

Use pixi when you want a self-contained environment with external netCDF tools
from conda-forge.

```bash
pixi install
pixi run test
pixi run lint
pixi run typecheck
pixi run smoke-python
pixi run smoke
```

The default pixi environment includes:

- Python runtime dependencies.
- NCO for `ncrcat`.
- `libnetcdf`, which provides the netCDF-C library tooling including `nccopy`.
- HDF5.
- pytest, ruff, and mypy.

Use the conda-forge package name `libnetcdf`. Do not use `netcdf-c`; it does not
solve on macOS.

The default pixi environment pins Python to 3.11 so `pixi run typecheck` matches
the package's minimum supported Python target. Revisit this when the project
raises its minimum Python version.

The `mpi` pixi environment is only a placeholder for future MPI work:

```bash
pixi run -e mpi mpi-info
```

## Backend Policy

- `python`: default. Requires only Python package dependencies and writes the
  final file with xarray/h5netcdf.
- `ncrcat`: optional. Requires NCO and is best supplied by pixi or HPC modules.
- `none`: writes validated `part_*.nc` files and a `zarr2nc-shards.json`
  manifest without creating a final output file.

Do not add CDO to v1 unless a concrete use case appears. It is another external
binary and does not improve the portable default path.

## Implementation Constraints

Keep these constraints unless there is a deliberate design change:

- The source is assumed to be written by `xarray.Dataset.to_zarr`.
- The source can be Zarr v2 or v3. Auto-detect by default through xarray; pass
  `zarr_format` only when the user explicitly asks to force a format.
- The default read mode is `decode_cf=False` to preserve encoded values and avoid
  accidental decode/re-encode changes.
- The normal writer uses `engine="h5netcdf"`.
- `netCDF4` is optional and not imported by normal commands.
- Non-MPI conversion uses independent one-file writes; do not make multiple
  processes write to the same HDF5 file.
- Shard manifests are authoritative for merge ordering. Do not rely on shell glob
  order in top-level workflows.
- Shard files should mark the split dimension as unlimited/record dimension so
  NCO `ncrcat` can concatenate them cleanly.
- Compression should default to conservative, widely readable HDF5/netCDF
  settings: gzip, shuffle, and a low compression level.

## Optional MPI Environment

The `zarr2nc-mpi` command is deliberately a placeholder. A real implementation
requires MPI-enabled HDF5 and netCDF-C, plus `netCDF4-python` built against that
stack.

Do not implement the MPI writer by calling `xarray.to_netcdf`. The intended data
path is:

1. Use xarray/zarr for metadata and source chunk reads.
2. Open the output with `netCDF4.Dataset(..., parallel=True, comm=...)`.
3. Create dimensions, variables, and attributes collectively on all ranks.
4. Assign non-overlapping hyperslabs to ranks.
5. Use collective writes where required, especially for compressed variables and
   unlimited-dimension appends.

## Testing Policy

Fast tests should not require external command-line tools. External tests must
skip cleanly when `ncrcat` or `nccopy` are unavailable.

Recommended split:

```bash
uv run pytest
uv run pytest -m "not external"
uv run pytest -m external
uv run pytest -m mpi
```

Required checks before release:

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

When pixi is available, also run:

```bash
pixi run test
pixi run smoke-python
pixi run smoke
```

## Next Tasks

1. Harden metadata normalization for calendars, strings, object arrays, and attrs
   that cannot be serialized by netCDF/HDF5.
2. Benchmark Python merge versus `ncrcat` across representative shard sizes and
   compression levels.
3. Add DataTree support with group-by-group conversion.
4. Add an explicit label-index merge option for files whose non-concatenation
   dimensions vary across parts, for example `--align-index nsite=sitenames`.
   The intended implementation is: rename the label variable to the dimension,
   promote it to an xarray index, concatenate with `join="outer"`, then reset
   the index and restore the label variable before writing NetCDF.
5. Add top-level `nccopy` repack options once there is a clear user-facing
   workflow for intermediate versus final output files.
6. Implement the MPI writer only in an MPI-capable pixi/conda or HPC module
   environment, not in the default `uv` development environment.
