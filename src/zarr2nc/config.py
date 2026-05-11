from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

NetcdfFormat = Literal["NETCDF4", "NETCDF4_CLASSIC"]
ZarrFormat = Literal[2, 3]


@dataclass(frozen=True)
class SliceOptions:
    """Options for writing one non-overlapping slice of a Zarr store to one netCDF file."""

    source: str
    target: str
    dim: str
    start: int
    stop: int
    group: str | None = None
    consolidated: bool | None = None
    zarr_format: ZarrFormat | None = None
    open_chunks: Any = "auto"
    output_chunks: dict[str, int] | None = None
    format: NetcdfFormat = "NETCDF4"
    compression: str | None = "gzip"
    complevel: int = 2
    shuffle: bool = True
    float_dtype: str | None = None
    unlimited_dims: tuple[str, ...] = ()
    drop_variables: tuple[str, ...] = ()
    decode_cf: bool = False
    scheduler: str = "threads"
    overwrite: bool = False
    atomic_write: bool = True

    @property
    def target_path(self) -> Path:
        return Path(self.target)
