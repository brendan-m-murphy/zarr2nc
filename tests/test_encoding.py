from __future__ import annotations

import pytest
import xarray as xr

from zarr2nc.encoding import (
    extract_fill_values,
    make_encoding,
    parse_consolidated,
    parse_csv,
    parse_dim_int_map,
    parse_open_chunks,
    parse_zarr_format,
)


def test_parse_csv() -> None:
    assert parse_csv(None) == ()
    assert parse_csv("time, member") == ("time", "member")


def test_parse_dim_int_map() -> None:
    assert parse_dim_int_map(None) is None
    assert parse_dim_int_map("time=24,lat=200") == {"time": 24, "lat": 200}


@pytest.mark.parametrize(
    ("text", "expected"),
    [("auto", None), ("true", True), ("false", False), ("1", True), ("0", False)],
)
def test_parse_consolidated(text: str, expected: bool | None) -> None:
    assert parse_consolidated(text) is expected


def test_parse_open_chunks() -> None:
    assert parse_open_chunks("auto") == "auto"
    assert parse_open_chunks("native") == {}
    assert parse_open_chunks("none") is None
    assert parse_open_chunks("time=10") == {"time": 10}


@pytest.mark.parametrize(("text", "expected"), [("auto", None), ("2", 2), ("3", 3)])
def test_parse_zarr_format(text: str, expected: int | None) -> None:
    assert parse_zarr_format(text) == expected


def test_extract_fill_values_moves_attr_to_encoding_input() -> None:
    ds = xr.Dataset({"x": ("time", [1.0, -999.0], {"_FillValue": -999.0})})

    out, fill_values = extract_fill_values(ds)

    assert "_FillValue" not in out["x"].attrs
    assert fill_values == {"x": -999.0}


def test_make_encoding_uses_chunks_compression_and_fill_value() -> None:
    ds = xr.Dataset({"x": (("time", "lat"), [[1.0, 2.0], [3.0, 4.0]])})

    encoding = make_encoding(
        ds,
        chunks={"time": 1, "lat": 5},
        compression="gzip",
        complevel=2,
        shuffle=True,
        float_dtype="float32",
        fill_values={"x": -999.0},
    )

    assert encoding["x"]["chunksizes"] == (1, 2)
    assert encoding["x"]["zlib"] is True
    assert encoding["x"]["complevel"] == 2
    assert encoding["x"]["dtype"] == "float32"
    assert encoding["x"]["_FillValue"] == -999.0
