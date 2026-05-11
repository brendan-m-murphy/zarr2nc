from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from zarr2nc.cli import main


def _write_source(path, *, zarr_format: int = 2) -> xr.Dataset:
    ds = xr.Dataset(
        {
            "foo": (("time", "lat"), np.arange(12, dtype="float32").reshape(6, 2)),
        },
        coords={"time": np.arange(6, dtype="int32"), "lat": np.array([50.0, 51.0])},
        attrs={"title": "cli test dataset"},
    )
    ds.to_zarr(path, zarr_format=zarr_format)
    return ds


@pytest.mark.integration
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_convert_uses_python_backend_by_default(tmp_path, zarr_format: int) -> None:
    pytest.importorskip("h5netcdf")
    pytest.importorskip("zarr")

    source = tmp_path / f"source-v{zarr_format}.zarr"
    target = tmp_path / f"merged-v{zarr_format}.nc"
    ds = _write_source(source, zarr_format=zarr_format)

    main(
        [
            "convert",
            str(source),
            str(target),
            "--dim",
            "time",
            "--shard-size",
            "2",
            "--jobs",
            "1",
            "--zarr-format",
            str(zarr_format),
            "--validate",
            "full",
            "--overwrite",
        ]
    )

    out = xr.open_dataset(target, engine="h5netcdf", decode_cf=False)
    try:
        assert out.sizes["time"] == 6
        np.testing.assert_array_equal(out["foo"].values, ds["foo"].values)
    finally:
        out.close()


@pytest.mark.integration
def test_convert_backend_none_keeps_part_files(tmp_path) -> None:
    pytest.importorskip("h5netcdf")
    pytest.importorskip("zarr")

    source = tmp_path / "source.zarr"
    target = tmp_path / "unused.nc"
    _write_source(source, zarr_format=2)

    main(
        [
            "convert",
            str(source),
            str(target),
            "--dim",
            "time",
            "--backend",
            "none",
            "--shard-size",
            "3",
            "--zarr-format",
            "2",
            "--validate",
            "quick",
            "--overwrite",
        ]
    )

    parts_dir = tmp_path / "unused_parts"
    assert not target.exists()
    assert (parts_dir / "zarr2nc-shards.json").exists()
    assert sorted(path.name for path in parts_dir.glob("part_*.nc")) == [
        "part_000000.nc",
        "part_000001.nc",
    ]
