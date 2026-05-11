from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from zarr2nc.config import SliceOptions
from zarr2nc.slice_writer import convert_slice


@pytest.mark.integration
def test_convert_slice_round_trips_basic_dataset(tmp_path) -> None:
    pytest.importorskip("h5netcdf")
    pytest.importorskip("zarr")

    source = tmp_path / "source.zarr"
    target = tmp_path / "part.nc"

    ds = xr.Dataset(
        {
            "foo": (("time", "lat"), np.arange(12, dtype="float32").reshape(6, 2)),
        },
        coords={"time": np.arange(6, dtype="int32"), "lat": np.array([50.0, 51.0])},
        attrs={"title": "minimal test dataset"},
    )
    ds.to_zarr(source, zarr_format=2)

    convert_slice(
        SliceOptions(
            source=str(source),
            target=str(target),
            dim="time",
            start=2,
            stop=5,
            zarr_format=2,
            output_chunks={"time": 2, "lat": 2},
            compression="gzip",
            complevel=1,
            overwrite=True,
        )
    )

    out = xr.open_dataset(target, engine="h5netcdf", decode_cf=False)

    assert out.sizes["time"] == 3
    np.testing.assert_array_equal(out["foo"].values, ds["foo"].isel(time=slice(2, 5)).values)
    np.testing.assert_array_equal(out["time"].values, np.array([2, 3, 4], dtype="int32"))
