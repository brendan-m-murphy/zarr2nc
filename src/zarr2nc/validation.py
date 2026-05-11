from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from zarr2nc.concat import parts_from_manifest, shards_from_manifest
from zarr2nc.config import ZarrFormat

ValidationMode = Literal["quick", "full"]


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


class ValidationError(ValueError):
    pass


def _open_source(
    source: str,
    *,
    group: str | None,
    consolidated: bool | None,
    zarr_format: ZarrFormat | None,
    decode_cf: bool,
) -> xr.Dataset:
    return xr.open_zarr(
        source,
        group=group,
        consolidated=consolidated,
        zarr_format=zarr_format,
        chunks=None,
        decode_cf=decode_cf,
        mask_and_scale=decode_cf,
        decode_times=decode_cf,
    )


def _compare_attr(expected: object, actual: object) -> bool:
    try:
        return bool(np.array_equal(np.asarray(expected), np.asarray(actual)))
    except Exception:
        return expected == actual


def _compare_array(name: str, expected: xr.DataArray, actual: xr.DataArray) -> str | None:
    expected_values = expected.values
    actual_values = actual.values
    try:
        if expected_values.dtype.kind in "biufc" and actual_values.dtype.kind in "biufc":
            np.testing.assert_allclose(expected_values, actual_values, equal_nan=True)
        else:
            np.testing.assert_array_equal(expected_values, actual_values)
    except AssertionError as exc:
        return f"{name!r} values differ: {exc}"
    return None


def compare_datasets(
    expected: xr.Dataset,
    actual: xr.Dataset,
    *,
    mode: ValidationMode,
) -> ValidationReport:
    errors: list[str] = []

    expected_sizes = dict(expected.sizes)
    actual_sizes = dict(actual.sizes)
    if expected_sizes != actual_sizes:
        errors.append(f"dimension sizes differ: expected {expected_sizes}, got {actual_sizes}")

    expected_vars = {str(name) for name in expected.variables}
    actual_vars = {str(name) for name in actual.variables}
    missing = sorted(expected_vars - actual_vars)
    extra = sorted(actual_vars - expected_vars)
    if missing:
        errors.append(f"missing variables: {missing}")
    if extra:
        errors.append(f"extra variables: {extra}")

    for name in sorted(expected_vars & actual_vars):
        expected_var = expected[name]
        actual_var = actual[name]
        if expected_var.dims != actual_var.dims:
            errors.append(
                f"{name!r} dimensions differ: expected {expected_var.dims}, got {actual_var.dims}"
            )
        if expected_var.dtype != actual_var.dtype:
            errors.append(
                f"{name!r} dtype differs: expected {expected_var.dtype}, got {actual_var.dtype}"
            )
        if mode == "full":
            error = _compare_array(name, expected_var, actual_var)
            if error:
                errors.append(error)

    for name, expected_value in expected.attrs.items():
        if name not in actual.attrs:
            errors.append(f"missing global attribute {name!r}")
        elif not _compare_attr(expected_value, actual.attrs[name]):
            errors.append(
                f"global attribute {name!r} differs: "
                f"expected {expected_value!r}, got {actual.attrs[name]!r}"
            )

    return ValidationReport(tuple(errors))


def raise_if_invalid(report: ValidationReport) -> None:
    if report.errors:
        joined = "\n".join(f"- {error}" for error in report.errors)
        raise ValidationError(f"validation failed:\n{joined}")


def validate_output(
    source: str,
    output: str | Path,
    *,
    dim: str,
    group: str | None = None,
    consolidated: bool | None = None,
    zarr_format: ZarrFormat | None = None,
    decode_cf: bool = False,
    mode: ValidationMode = "quick",
) -> ValidationReport:
    source_ds = _open_source(
        source,
        group=group,
        consolidated=consolidated,
        zarr_format=zarr_format,
        decode_cf=decode_cf,
    )
    output_ds = xr.open_dataset(output, engine="h5netcdf", decode_cf=decode_cf)
    try:
        if dim in source_ds.sizes:
            stop = output_ds.sizes.get(dim, source_ds.sizes[dim])
            source_ds = source_ds.isel({dim: slice(0, stop)})
        return compare_datasets(source_ds, output_ds, mode=mode)
    finally:
        source_ds.close()
        output_ds.close()


def validate_parts(
    source: str,
    manifest_path: str | Path,
    *,
    group: str | None = None,
    consolidated: bool | None = None,
    zarr_format: ZarrFormat | None = None,
    decode_cf: bool = False,
    mode: ValidationMode = "quick",
) -> ValidationReport:
    source_ds = _open_source(
        source,
        group=group,
        consolidated=consolidated,
        zarr_format=zarr_format,
        decode_cf=decode_cf,
    )
    errors: list[str] = []
    try:
        shard_parts = parts_from_manifest(manifest_path)
        for shard, part in zip(shards_from_manifest(manifest_path), shard_parts, strict=True):
            dim = shard["dim"]
            expected = source_ds.isel({dim: slice(shard["start"], shard["stop"])})
            actual = xr.open_dataset(part, engine="h5netcdf", decode_cf=decode_cf)
            try:
                report = compare_datasets(expected, actual, mode=mode)
                errors.extend(f"{Path(part).name}: {error}" for error in report.errors)
            finally:
                actual.close()
    finally:
        source_ds.close()
    return ValidationReport(tuple(errors))


def format_report(report: ValidationReport) -> str:
    if report.ok:
        return "validation ok"
    return "\n".join(("validation failed", *(f"- {error}" for error in report.errors)))


def combine_reports(reports: Iterable[ValidationReport]) -> ValidationReport:
    errors: list[str] = []
    for report in reports:
        errors.extend(report.errors)
    return ValidationReport(tuple(errors))
