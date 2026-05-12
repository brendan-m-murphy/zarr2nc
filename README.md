# zarr2nc

CLI tools for exporting **xarray-created Zarr stores** to collaborator-facing
netCDF4/HDF5 files.

The default path is portable Python:

```text
xarray Dataset.to_zarr store
        -> zarr2nc convert writes independent shard files in parallel
        -> zarr2nc convert merges shards with xarray/h5netcdf
        -> final .nc file
```

External netCDF tools are optional. If `ncrcat` and `nccopy` are available, they
can be used for faster concatenation and repacking, but normal conversion does
not require NCO, CDO, MPI, netCDF4-python, or preloaded HPC modules.

## Install

Use the Python-only tool path when portability matters:

```bash
uv tool install .
zarr2nc doctor
```

For a self-contained environment with NCO, the netCDF-C library tooling, and HDF5
from conda-forge:

```bash
pixi install
pixi run zarr2nc doctor
```

## Convert

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
format. To force a specific source format, pass `--zarr-format 2` or
`--zarr-format 3`. The converter reads both Zarr v2 and v3 stores through
`xarray.open_zarr`.

When `--jobs` is greater than 1 and no shard policy is supplied, `zarr2nc convert`
creates one shard per worker. You can control this explicitly:

```bash
zarr2nc convert input.zarr output.nc --dim time --shard-size 744 --jobs 8
zarr2nc convert input.zarr output.nc --dim time --num-shards 16 --jobs 8
```

Optional NCO merge backend:

```bash
pixi run zarr2nc convert input.zarr output.nc \
  --dim time \
  --jobs 8 \
  --backend ncrcat \
  --validate full \
  --overwrite
```

Keep part files instead of creating a final file:

```bash
zarr2nc convert input.zarr output.nc \
  --dim time \
  --backend none \
  --shard-size 744 \
  --validate quick \
  --overwrite
```

This writes `output_parts/part_*.nc` and `output_parts/zarr2nc-shards.json`.

## Low-Level Commands

The original commands remain available for manual workflows:

```bash
zarr2nc-slice input.zarr parts/part_000000.nc --dim time --start 0 --stop 744
zarr2nc-shards input.zarr parts --dim time --shard-size 744 --jobs 8
zarr2nc-concat --manifest parts/zarr2nc-shards.json --output merged.nc --overwrite
```

`zarr2nc-concat` uses `ncrcat`, so it requires NCO. The top-level `zarr2nc convert`
command uses the Python merge backend by default.

## Development

Python-only development:

```bash
uv venv
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run mypy src
```

Pixi development with external netCDF tools:

```bash
pixi run test
pixi run lint
pixi run typecheck
pixi run smoke-python
pixi run smoke
```

The pixi spec uses the conda-forge package `libnetcdf` for netCDF-C and `nccopy`.
Do not replace it with `netcdf-c`; that package name does not solve on macOS.
The default pixi environment currently pins Python to 3.11 so its type-checking
target matches the package's minimum supported Python version.

The example Zarr generator writes Zarr v2 by default:

```bash
python examples/create_example_zarr.py
python examples/create_example_zarr.py --zarr-format 3 --output example-v3.zarr
```

## Current Scope

Supported first:

- Zarr stores created by `xarray.Dataset.to_zarr`.
- Zarr v2 and v3 input stores, auto-detected by default.
- One main split dimension, usually `time`.
- Dataset-level conversion.
- File-level parallelism with one process per output shard.
- Python-only final merge through `xarray` and `h5netcdf`.
- Optional NCO final merge through `ncrcat`.

Deferred:

- Full `DataTree` support.
- True MPI writes into one shared netCDF4/HDF5 file.
- Raw compressed Zarr chunk transfer into HDF5 chunks.
- Arbitrary third-party Zarr stores without xarray dimension metadata.
- Direct support for merging parts whose non-concatenation dimensions change
  size across files. The Python backend now reports this clearly. A likely
  future interface is an explicit label-index option such as
  `--align-index nsite=sitenames`, which would promote `sitenames` to an xarray
  index for merge alignment, then reset it before writing NetCDF.
