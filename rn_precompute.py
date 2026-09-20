"""Erzeugt die vorberechneten Mechanismen `weights/*.npz` und `weights/manifest.json` (Trainingsdauer, Test-Kennzahlen).
Aufruf: `python rn_precompute.py [name ...]` (ohne Namen: alle). Trainings-Seed 0; Test-Stichproben sind die festen Seeds ab 100000."""

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import cn_constants as C
from rn_evaluation import evaluate
from rn_mechanism import save_mechanism
from rn_train import DEFAULT_A, DEFAULT_B, train
from rn_types import WorldA, WorldB

WEIGHTS = Path(__file__).resolve().parent / "weights"
N_TEST_B, N_TEST_A = 200, 100


def build(name):
    kind, n, k, description, overrides = C.WEIGHT_SPECS[name]
    world = WorldB(n, k) if kind == "B" else WorldA(n, k)
    cfg = replace(DEFAULT_B if kind == "B" else DEFAULT_A, **overrides)
    result = train(world, cfg)
    reports = world.heldout(N_TEST_B if kind == "B" else N_TEST_A)
    metrics = evaluate(world, result.mechanism, reports, np.random.default_rng(1), "a_ascent", result.mechanism)
    meta = {"name": name, "description": description, "train_seconds": round(result.seconds, 1),
            "config": {key: getattr(cfg, key) for key in cfg.__dataclass_fields__}, "metrics": metrics,
            "history": {key: [round(v, 4) for v in values] for key, values in result.history.items()}}
    save_mechanism(WEIGHTS / f"{name}.npz", world, result.mechanism.net, meta)
    print(name, round(result.seconds, 1), {m: round(v, 3) if isinstance(v, float) else v for m, v in metrics.items()}, flush=True)
    return meta


def build_wall():
    """Die Wand: Lücke eines Allokations-Netzes OHNE jede Anreiz-Nebenbedingung (nur Makespan minimieren) je (n, k), Typ-A-Instanzen;
    daneben die vorberechneten gelernten Mechanismen (Lücke, Regret)."""
    rows = []
    manifest = {}
    for path in sorted(WEIGHTS.glob("*.npz")):
        meta = json.loads(str(np.load(path, allow_pickle=False)["meta"]))
        manifest[meta["name"]] = meta
    for n, k in ((3, 2), (3, 3), (4, 2), (4, 3)):
        world = WorldA(n, k)
        cfg = replace(DEFAULT_A, steps=3000, warmup=3000, kappa_max=0.0, lr0=0.005, lr1=0.0005)     # warmup = steps: nie ein Regret-Term
        result = train(world, cfg)
        reports = world.heldout(N_TEST_A)
        gap = evaluate_gap(world, result.mechanism, reports)
        learned = next((m for m in manifest.values() if (m["kind"], m["n"], m["k"]) == ("A", n, k) and m["name"].endswith("_mid")), None)
        rows.append({"n": n, "k": k, "plain_gap": gap, "seconds": round(result.seconds, 1),
                     "learned_gap": learned["metrics"]["gap_pct"] if learned else None,
                     "learned_regret": learned["metrics"]["regret"] if learned else None})
        print("wall", n, k, rows[-1], flush=True)
    (WEIGHTS / "wall.json").write_text(json.dumps({"plain": rows, "steps": 3000}, indent=1), encoding="utf-8")


def evaluate_gap(world, mech, reports):
    from rn_evaluation import basic_metrics
    return float(basic_metrics(world, mech, reports)["gap_pct"])


def write_manifest():
    manifest = {}
    for path in sorted(WEIGHTS.glob("*.npz")):
        data = np.load(path, allow_pickle=False)
        manifest[path.stem] = json.loads(str(data["meta"]))
    (WEIGHTS / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    names = sys.argv[1:] or list(C.WEIGHT_SPECS)
    WEIGHTS.mkdir(exist_ok=True)
    if names == ["wall"]:
        build_wall()
    else:
        for name in names:
            build(name)
    write_manifest()
