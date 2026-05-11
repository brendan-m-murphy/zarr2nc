from __future__ import annotations

import shutil

import numpy as np
import pytest
import xarray as xr

from zarr2nc.cli import main
from zarr2nc.concat import repack_file


def _write_source(path) -> xr.Dataset:
    ds = xr.Dataset(
        {
            "foo": (("time", "lat"), np.arange(12, dtype="float32").reshape(6, 2)),
        },
        coords={"time": np.arange(6, dtype="int32"), "lat": np.array([50.0, 51.0])},
    )
    ds.to_zarr(path, zarr_format=2)
    return ds


@pytest.mark.external
@pytest.mark.integration
def test_convert_with_ncrcat_backend_when_available(tmp_path) -> None:
    pytest.importorskip("h5netcdf")
    pytest.importorskip("zarr")
    if shutil.which("ncrcat") is None:
        pytest.skip("ncrcat is not available")

    source = tmp_path / "source.zarr"
    target = tmp_path / "merged.nc"
    ds = _write_source(source)

    main(
        [
            "convert",
            str(source),
            str(target),
            "--dim",
            "time",
            "--shard-size",
            "3",
            "--backend",
            "ncrcat",
            "--validate",
            "full",
            "--overwrite",
        ]
    )

    out = xr.open_dataset(target, engine="h5netcdf", decode_cf=False)
    try:
        np.testing.assert_array_equal(out["foo"].values, ds["foo"].values)
    finally:
        out.close()


@pytest.mark.external
@pytest.mark.integration
def test_repack_with_nccopy_when_available(tmp_path) -> None:
    pytest.importorskip("h5netcdf")
    if shutil.which("nccopy") is None:
        pytest.skip("nccopy is not available")

    source = tmp_path / "source.nc"
    target = tmp_path / "target.nc"
    ds = xr.Dataset({"foo": ("time", np.arange(4, dtype="float32"))})
    ds.to_netcdf(source, engine="h5netcdf")

    repack_file(str(source), str(target), deflate=1, shuffle=True)

    assert target.exists()
