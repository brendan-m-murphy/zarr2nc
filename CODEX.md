# CODEX.md

## Objective

This repository is a skeleton for a standalone converter from **xarray-flavour Zarr**
to collaborator-facing netCDF4/HDF5 products.

Primary path:

```text
xarray Dataset.to_zarr store
  -> zarr2nc-shards writes many netCDF shard files in parallel
  -> zarr2nc-concat wraps ncrcat and optional nccopy
  -> final .nc file, or a multi-file part_*.nc deliverable
```

The normal implementation should avoid `netCDF4` as a hard dependency. Use
`h5netcdf`/`h5py` for the simple writer path. `netCDF4`, MPI-enabled HDF5, and
MPI-enabled netCDF-C belong in optional integration environments only.

## Repository layout

```text
src/zarr2nc/encoding.py       CLI parsing helpers and netCDF encoding policy
src/zarr2nc/slice_writer.py   zarr2nc-slice implementation
src/zarr2nc/shards.py         zarr2nc-shards implementation and shard planning
src/zarr2nc/concat.py         ncrcat/nccopy command wrappers
src/zarr2nc/mpi.py            placeholder for true MPI single-file writer
tests/                        unit and lightweight integration tests
```

## Development environment: uv path

Use this path for normal coding and tests.

```bash
uv venv
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run mypy src
```

Run one integration-style smoke test manually:

```bash
uv run pytest -m integration
```

The current integration test creates a tiny xarray dataset, writes it to Zarr,
converts one slice to `.nc` using `h5netcdf`, and reopens the result.

## Development environment: conda/mamba path

Use this path when external netCDF tooling or `netCDF4` is needed. The motivation
is that `netCDF4` and `h5py` coexist much more reliably when installed from
conda-forge, but the main codebase should remain easy to develop with `uv`.

```bash
mamba create -n zarr2nc-netcdf -c conda-forge \
  python=3.12 \
  xarray zarr dask h5netcdf h5py netcdf4 \
  nco netcdf-c \
  pytest ruff mypy

mamba activate zarr2nc-netcdf
python -m pip install -e . --no-deps
pytest
ruff check .
mypy src
```

Check that external tools are visible:

```bash
which ncrcat
which nccopy
ncrcat --version
nccopy -h 2>&1 | head
```

## Optional MPI environment

The `zarr2nc-mpi` command is deliberately a placeholder. A real implementation
requires MPI-enabled HDF5 and netCDF-C, plus `netCDF4-python` built against that
stack.

Sketch environment, to be refined on the actual HPC system:

```bash
mamba create -n zarr2nc-mpi -c conda-forge \
  python=3.12 \
  xarray zarr dask h5netcdf h5py \
  mpi4py netcdf4 \
  "hdf5=*=mpi*" "netcdf-c=*=mpi*" \
  pytest

mamba activate zarr2nc-mpi
python -m pip install -e . --no-deps
mpiexec -n 4 zarr2nc-mpi input.zarr output.nc --dim time
```

Do not implement the MPI writer by calling `xarray.to_netcdf`. The intended data
path is:

1. Use xarray/zarr for metadata and source chunk reads.
2. Open the output with `netCDF4.Dataset(..., parallel=True, comm=...)`.
3. Create dimensions, variables, and attributes collectively on all ranks.
4. Assign non-overlapping hyperslabs to ranks.
5. Use collective writes where required, especially for compressed variables and
   unlimited-dimension appends.

## Manual smoke-test plan

Create a test Zarr store:

```bash
uv run python - <<'PY'
from pathlib import Path
import numpy as np
import xarray as xr

out = Path('example.zarr')
if out.exists():
    import shutil
    shutil.rmtree(out)

ds = xr.Dataset(
    {
        'foo': (('time', 'lat', 'lon'), np.random.default_rng(0).normal(size=(48, 3, 4)).astype('float32')),
    },
    coords={
        'time': np.arange(48, dtype='int32'),
        'lat': np.linspace(50, 52, 3),
        'lon': np.linspace(-5, -2, 4),
    },
)
ds.to_zarr(out)
PY
```

Write shards:

```bash
uv run zarr2nc-shards example.zarr parts \
  --dim time \
  --shard-size 12 \
  --jobs 4 \
  --chunks time=6,lat=3,lon=4 \
  --compression gzip \
  --complevel 1 \
  --unlimited-dim time \
  --overwrite
```

Concatenate if NCO is available:

```bash
zarr2nc-concat 'parts/part_*.nc' --output merged.nc --overwrite
```

Repack if `nccopy` is available:

```bash
zarr2nc-concat 'parts/part_*.nc' \
  --output merged.nc \
  --repack-output final.nc \
  --nccopy-deflate 2 \
  --nccopy-shuffle \
  --nccopy-chunks time=24,lat=3,lon=4 \
  --overwrite
```

## Testing policy

Keep fast unit tests independent from external command-line tools.

Recommended split:

```bash
uv run pytest                         # unit + lightweight h5netcdf integration
uv run pytest -m 'not external'       # no ncrcat/nccopy execution
uv run pytest -m external             # requires ncrcat/nccopy
uv run pytest -m mpi                  # requires MPI-enabled netCDF/HDF5
```

Current tests:

- `test_encoding.py`: parsing and encoding policies.
- `test_shards.py`: shard planning.
- `test_concat.py`: ncrcat/nccopy command construction only.
- `test_slice_writer.py`: tiny real Zarr -> `.nc` conversion using `h5netcdf`.

## Implementation constraints

Keep these constraints unless there is a deliberate design change:

- The source is assumed to be written by `xarray.Dataset.to_zarr`.
- The default read mode is `decode_cf=False` to preserve encoded values and avoid
  accidental decode/re-encode changes.
- The normal writer uses `engine="h5netcdf"`.
- `netCDF4` is optional and not imported by normal commands.
- Shard writes should be independent one-file writes. Do not make multiple
  processes write to the same HDF5 file in the non-MPI path.
- Shard files should mark the split dimension as unlimited/record dimension so
  NCO `ncrcat` can concatenate them cleanly.
- Compression should default to conservative, widely readable HDF5/netCDF
  settings: gzip, shuffle, low compression level.

## Next tasks for Codex

1. Add external-tool integration tests that are skipped when `ncrcat` or `nccopy`
   are unavailable.
2. Add a validation command that compares source Zarr slices against produced
   `.nc` files using xarray.
3. Add `DataTree` support with group-by-group conversion.
4. Add a manifest-driven concat path that reads `zarr2nc-shards.json` rather than
   relying on shell glob ordering.
5. Add richer metadata normalization for calendars, `_FillValue`, object arrays,
   strings, and attrs that cannot be serialized by netCDF/HDF5.
6. Benchmark shard sizes and compression levels against a representative dataset.
7. Implement the MPI writer only in the conda/mamba MPI environment, not in the
   default `uv` development environment.
