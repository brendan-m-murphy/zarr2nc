from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from zarr2nc.merge import MergeAlignmentError, merge_python_parts


def _write_concentration_part(path: Path, *, year: int, site_names: list[str]) -> None:
    nsite = len(site_names)
    data = np.arange(2 * nsite, dtype="float64").reshape(2, nsite) + year
    ds = xr.Dataset(
        {
            "sitenames": (
                "nsite",
                np.asarray(site_names, dtype=object),
                {"long_name": "identifier of site"},
            ),
            "Yobs": (
                ("time", "nsite"),
                data,
                {
                    "units": "mol mol-1",
                    "long_name": "observed_mole_fraction",
                    "coordinates": "sitenames",
                },
            ),
        },
        coords={"time": np.arange(2) + (year - 2013) * 2},
    )
    ds.to_netcdf(path, engine="h5netcdf", unlimited_dims=("time",))


def _merge_with_site_index(parts: list[Path], output: Path) -> None:
    datasets: list[xr.Dataset] = []
    try:
        for part in parts:
            ds = xr.open_dataset(part, engine="h5netcdf", decode_cf=False)
            ds = ds.rename_vars({"sitenames": "nsite"})
            ds = ds.set_coords("nsite")
            ds = ds.set_xindex("nsite")
            datasets.append(ds)

        merged = xr.concat(datasets, dim="time", join="outer")
        merged = merged.reset_index("nsite").rename_vars({"nsite": "sitenames"})
        merged.to_netcdf(output, engine="h5netcdf", unlimited_dims=("time",))
    finally:
        for ds in datasets:
            ds.close()


@pytest.fixture
def varying_site_parts(tmp_path) -> list[Path]:
    parts = [tmp_path / "SF6_EUROPE_PARIS_conc_2013-01-01.nc", tmp_path / "2014.nc"]
    _write_concentration_part(parts[0], year=2013, site_names=[f"S{i}" for i in range(6)])
    _write_concentration_part(parts[1], year=2014, site_names=[f"S{i}" for i in range(8)])
    return parts


def test_python_merge_rejects_varying_unindexed_site_dimension(
    tmp_path,
    varying_site_parts: list[Path],
) -> None:
    with pytest.raises(MergeAlignmentError, match="sitenames\\(nsite\\)"):
        merge_python_parts(
            [str(path) for path in varying_site_parts],
            tmp_path / "merged.nc",
            dim="time",
            overwrite=True,
        )


def test_varying_site_dimension_can_be_merged_with_xarray_index_workaround(
    tmp_path,
    varying_site_parts: list[Path],
) -> None:
    output = tmp_path / "merged-with-site-index.nc"

    _merge_with_site_index(varying_site_parts, output)

    merged = xr.open_dataset(output, engine="h5netcdf", decode_cf=False)
    try:
        assert merged.sizes == {"time": 4, "nsite": 8}
        assert merged["sitenames"].values.tolist() == [f"S{i}" for i in range(8)]
        np.testing.assert_allclose(merged["Yobs"].isel(time=slice(0, 2), nsite=slice(6, 8)), np.nan)
        np.testing.assert_allclose(
            merged["Yobs"].isel(time=slice(2, 4), nsite=slice(6, 8)),
            np.asarray([[2020.0, 2021.0], [2028.0, 2029.0]]),
        )
    finally:
        merged.close()
