"""Jedes Preset zeigt, was sein Name und seine Hilfe behaupten (Bänder mit dem ausgelieferten Code kalibriert)."""

from dataclasses import replace
from pathlib import Path

import pytest

import cn_constants as C
from rn_evaluation import comparison_table, verdict
from rn_mechanism import load_mechanism
from rn_train import QUICK_B, TrainConfig, train
from rn_types import WorldA, WorldB

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"


def _measure(p):
    if p["source"] == "live":
        world = WorldB(p["n"], p["k"])
        cfg = replace(QUICK_B, steps=p["steps"], mu=p["mu"], kappa_max=p["kappa"], seed=p["lseed"])
        learned = train(world, cfg).mechanism
    else:
        learned, world, _ = load_mechanism(WEIGHTS / f"{p['source']}.npz")
    rows = comparison_table(world, world.heldout(C.N_TEST_B if world.kind == "B" else C.N_TEST_A), learned, gamma=p["gamma"])
    by = {r["key"]: r for r in rows}
    out = {"verdict": verdict(rows)[1]}
    for key, prefix in (("learned", "learned"), ("learned_rounded", "rounded"), ("A", "A"), ("B", "B"), ("boost", "boost"), ("threshold", "threshold")):
        if key in by:
            out[f"{prefix}_gap"], out[f"{prefix}_pay"], out[f"{prefix}_regret"] = by[key]["gap_pct"], by[key]["pay_cost"], by[key]["regret"]
    return out


def test_every_preset_has_help_and_bands():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS)
    assert len(C.PRESETS) == 8


def test_presets_use_supported_combinations():
    for p in C.PRESETS.values():
        assert p["kind"] in ("A", "B") and p["k"] in (2, 3)
        if p["source"] != "live":
            assert (p["kind"], p["n"], p["k"]) == C.WEIGHT_SPECS[p["source"]][:3]
            assert (WEIGHTS / f"{p['source']}.npz").exists()
        else:
            assert p["kind"] == "B" and p["k"] ** p["n"] <= C.LIVE_MAX_PARTITIONS


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_stays_inside_its_bands(name):
    measured = _measure(C.PRESETS[name])
    for key, expected in C.PRESET_EXPECTED_BANDS[name].items():
        value = measured[key]
        if isinstance(expected, str):
            assert value == expected, f"{key}: {value}"
        else:
            lo, hi = expected
            assert lo <= value <= hi, f"{key}: {value} nicht in [{lo}, {hi}]"
