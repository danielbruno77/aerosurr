from __future__ import annotations

import json
from pathlib import Path

import pytest

from aerosurr.data.splits import all_runs, load_manifest, priority_runs, run_number


def test_run_number() -> None:
    assert run_number("run_473") == 473
    with pytest.raises(ValueError):
        run_number("run473")


def test_priority_runs_dedupes_in_order() -> None:
    m = {"a": ["run_3", "run_1"], "b": ["run_1", "run_2"], "c": []}
    assert priority_runs(m, ["b", "a", "c"]) == ["run_1", "run_2", "run_3"]
    with pytest.raises(KeyError, match="'zzz'"):
        priority_runs(m, ["zzz"])


def test_load_manifest_validates(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    p.write_text(
        json.dumps({"full_train": ["run_2", "run_10"], "full_val": [], "full_test": ["run_1"]})
    )
    m = load_manifest(p)
    assert all_runs(m) == ["run_1", "run_2", "run_10"]
    p.write_text(json.dumps({"full_train": "run_1"}))
    with pytest.raises(ValueError):
        load_manifest(p)
