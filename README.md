# zarr2nc

Skeleton CLI tools for exporting **xarray-created Zarr stores** to collaborator-facing
netCDF4/HDF5 files.

The design goal is to keep HDF5/netCDF packaging in a separate executable environment,
so the main analysis stack can keep using Zarr without carrying a fragile netCDF4/HDF5
binary stack.

## Intended workflow

```text
xarray.Dataset.to_zarr(...)
        ↓
zarr2nc-shards: write many part_*.nc files in parallel
        ↓
zarr2nc-concat: ncrcat part_*.nc into merged.nc
        ↓
optional nccopy repack/rechunk/compress into final.nc
```

The default implementation writes through `xarray.to_netcdf(engine="h5netcdf")` and
therefore avoids a runtime dependency on `netCDF4` in the normal development path.

## Commands

### Write one slice

```bash
zarr2nc-slice input.zarr parts/part_000000.nc \
  --dim time \
  --start 0 \
  --stop 744 \
  --chunks time=24,lat=200,lon=200 \
  --compression gzip \
  --complevel 1 \
  --unlimited-dim time \
  --overwrite
```

### Write shards in parallel

```bash
zarr2nc-shards input.zarr parts \
  --dim time \
  --shard-size 744 \
  --jobs 8 \
  --chunks time=24,lat=200,lon=200 \
  --compression gzip \
  --complevel 1 \
  --unlimited-dim time \
  --overwrite
```

Each process writes a separate `.nc` file. This is file-level parallelism, not
parallel writes into one HDF5 file.

### Concatenate shards

Requires NCO (`ncrcat`) and, for repacking, netCDF-C (`nccopy`).

```bash
zarr2nc-concat 'parts/part_*.nc' \
  --output merged.nc \
  --overwrite
```

Optional final repack:

```bash
zarr2nc-concat 'parts/part_*.nc' \
  --output merged.nc \
  --repack-output final.nc \
  --nccopy-deflate 2 \
  --nccopy-shuffle \
  --nccopy-chunks time=24,lat=200,lon=200 \
  --overwrite
```

## Current scope

Supported first:

- Zarr stores created by `xarray.Dataset.to_zarr`.
- One main split dimension, usually `time`.
- Dataset-level conversion.
- HDF5/netCDF writing through `h5netcdf`.
- External concatenation through NCO `ncrcat`.

Not implemented yet:

- Full `DataTree` support.
- True MPI writes into one netCDF4/HDF5 file.
- Raw compressed Zarr chunk transfer into HDF5 chunks.
- Arbitrary third-party Zarr stores without xarray dimension metadata.

## Development setup with uv

```bash
uv venv
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run mypy src
```

For the normal non-MPI path, do not add `netCDF4` as a hard dependency. The point
of this package is to let development stay in a lightweight `uv` environment while
the packaging tool lives independently from the main analysis environment.
