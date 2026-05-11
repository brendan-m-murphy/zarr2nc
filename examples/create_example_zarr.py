from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

parser = argparse.ArgumentParser(description="Create a tiny example xarray Zarr store.")
parser.add_argument("--output", default="example.zarr")
parser.add_argument("--zarr-format", type=int, choices=[2, 3], default=2)
args = parser.parse_args()

out = Path(args.output)
if out.exists():
    shutil.rmtree(out)

rng = np.random.default_rng(0)
ds = xr.Dataset(
    {
        "foo": (
            ("time", "lat", "lon"),
            rng.normal(size=(48, 3, 4)).astype("float32"),
        ),
    },
    coords={
        "time": np.arange(48, dtype="int32"),
        "lat": np.linspace(50, 52, 3),
        "lon": np.linspace(-5, -2, 4),
    },
    attrs={"title": "example xarray-flavour Zarr store"},
)

ds.to_zarr(out, zarr_format=args.zarr_format)
print(out)
