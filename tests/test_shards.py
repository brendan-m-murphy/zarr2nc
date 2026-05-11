from __future__ import annotations

import pytest

from zarr2nc.shards import plan_shards


def test_plan_shards_by_size() -> None:
    assert plan_shards(10, shard_size=4) == [(0, 4), (4, 8), (8, 10)]


def test_plan_shards_by_count() -> None:
    assert plan_shards(10, num_shards=3) == [(0, 3), (3, 6), (6, 10)]


def test_plan_shards_default_single_shard() -> None:
    assert plan_shards(10) == [(0, 10)]


def test_plan_shards_rejects_ambiguous_split_policy() -> None:
    with pytest.raises(ValueError, match="only one"):
        plan_shards(10, shard_size=2, num_shards=3)
