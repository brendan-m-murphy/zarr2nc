from __future__ import annotations

from zarr2nc.concat import (
    build_nccopy_command,
    build_ncrcat_command,
    expand_parts,
    parts_from_manifest,
)


def test_build_ncrcat_command() -> None:
    cmd = build_ncrcat_command(["part_000.nc", "part_001.nc"], "merged.nc", overwrite=True)
    assert cmd == ["ncrcat", "-O", "-4", "part_000.nc", "part_001.nc", "merged.nc"]


def test_build_nccopy_command() -> None:
    cmd = build_nccopy_command(
        "merged.nc",
        "final.nc",
        chunks={"time": 24, "lat": 200},
        deflate=2,
        shuffle=True,
        in_memory=True,
    )
    assert cmd == [
        "nccopy",
        "-k",
        "nc4",
        "-d",
        "2",
        "-s",
        "-c",
        "time/24,lat/200",
        "-w",
        "merged.nc",
        "final.nc",
    ]


def test_expand_parts_keeps_non_matching_items() -> None:
    assert expand_parts(["definitely-not-a-real-file-*.nc"]) == ["definitely-not-a-real-file-*.nc"]


def test_parts_from_manifest_sorts_by_offsets(tmp_path) -> None:
    manifest = tmp_path / "zarr2nc-shards.json"
    manifest.write_text(
        """
        {
          "version": 1,
          "shards": [
            {"path": "part_001.nc", "dim": "time", "start": 2, "stop": 4},
            {"path": "part_000.nc", "dim": "time", "start": 0, "stop": 2}
          ]
        }
        """
    )

    assert parts_from_manifest(manifest) == [
        str(tmp_path / "part_000.nc"),
        str(tmp_path / "part_001.nc"),
    ]
