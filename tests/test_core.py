import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import cn_constants as C
from auction_bids import bundle_bid_table
from auction_mechanism import run_mechanism
from cn_scenario import generate_instance
from rn_baselines import hand_mechanism, threshold_mechanism, utilities, work_curve, work_steps
from rn_evaluation import comparison_table, hand, threshold_regret_sample
from rn_mechanism import LearnedMechanism, load_mechanism, make_net, save_mechanism
from rn_partitions import partitions
from rn_regret import regret_type_a_hillclimb, regret_type_a_lambda_grid, regret_type_b
from rn_train import QUICK_B, TrainConfig, surrogate, train
from rn_types import C_HI, C_LO, HELDOUT_SEED_BASE, WorldA, WorldB, bundle_bids_batch

ROOT = Path(__file__).resolve().parent.parent


# --- Partitionen und Welten -----------------------------------------------------------------------

def test_partitions_cover_every_job_exactly_once():
    for n, k in ((2, 2), (3, 3), (4, 2)):
        masks, sizes = partitions(n, k)
        assert masks.shape == (k ** n, k)
        for row in masks:
            union = 0
            for m in row:
                assert union & m == 0
                union |= int(m)
            assert union == (1 << n) - 1
        assert (sizes.sum(1) == n).all()
        assert len({tuple(r) for r in masks}) == k ** n                              # alle verschieden


def test_costs_of_world_a_match_bundle_bid_table():
    world = WorldA(3, 2)
    bids = world.heldout(3)
    costs = world.costs(bids)
    masks, _ = partitions(3, 2)
    for i in range(3):
        for p in range(world.P):
            for a in range(2):
                assert costs[i, p, a] == bids[i, a, masks[p, a]]
    assert np.allclose(bids[0], bundle_bid_table(generate_instance(3, 2, 0.3, 1.0, HELDOUT_SEED_BASE)))


def test_vectorised_training_generator_matches_real_generator_for_a_fixed_seed():
    inst = generate_instance(4, 3, 0.3, 1.0, 5)
    pos = np.array([[j.position for j in inst.jobs]])
    dur = np.array([[j.duration for j in inst.jobs]])
    batch = bundle_bids_batch(pos, dur, np.array(inst.agent_start_positions), 1.0)
    assert np.allclose(batch[0], bundle_bid_table(inst))


# --- Handregeln gegen auction-demo --------------------------------------------------------------------

@pytest.mark.parametrize("rule", ["A", "B", "E"])
def test_hand_rules_match_auction_demo_run_mechanism(rule):
    world = WorldA(4, 3)
    for seed in range(6):
        inst = generate_instance(4, 3, 0.3, 1.0, HELDOUT_SEED_BASE + seed)
        bids = bundle_bid_table(inst)
        pi, pay = hand_mechanism(world, bids[None], rule)
        out = run_mechanism(bids, rule)
        masks = tuple(int(m) for m in world.masks[int(pi.argmax(1)[0])])
        if masks == tuple(out.masks):                                                 # gleiche Zuteilung => gleiche Zahlungen
            assert np.allclose(pay[0], out.payments, atol=1e-9)
        if rule != "B":
            assert abs(world.costs(bids[None])[0][int(pi.argmax(1)[0])].max() - out.value) < 1e-9


def test_vcg_and_boosted_vcg_have_zero_regret_and_ir():
    world = WorldA(3, 2)
    bids = world.heldout(20)
    rng = np.random.default_rng(0)
    for rule, gamma in (("B", 0.0), ("boost", 3.0), ("boost", 8.0)):
        mech = hand(rule, gamma)
        assert regret_type_a_lambda_grid(world, mech, bids).max() < 1e-9
        assert regret_type_a_hillclimb(world, mech, bids, rng, restarts=3, iters=30).max() < 1e-9
        assert utilities(world, mech, bids).min() > -1e-9                             # individuell rational


def test_pay_as_bid_is_manipulable_and_clarke_on_makespan_violates_ir():
    world = WorldA(3, 2)
    bids = world.heldout(30)
    assert regret_type_a_lambda_grid(world, hand("A"), bids).mean() > 10
    assert (utilities(world, hand("E"), bids) < -1e-9).any()


def test_boost_gamma_zero_is_vcg():
    world = WorldA(3, 3)
    bids = world.heldout(10)
    pi_b, pay_b = hand_mechanism(world, bids, "B")
    pi_g, pay_g = hand_mechanism(world, bids, "boost", 0.0)
    assert np.allclose(pi_b, pi_g) and np.allclose(pay_b, pay_g)


# --- Ein Parameter: Handbeispiel und Wahrhaftigkeit -----------------------------------------------------

def test_hand_example_thresholds_payment_and_utility():
    world = WorldB(2, 2)
    c = np.array([[1.2, 1.0]])
    pi, pay = threshold_mechanism(world, c)
    p = int(pi.argmax(1)[0])
    assert sorted(int(m) for m in world.masks[p]) == [1, 2]                        # jeder Agent einen Auftrag
    assert abs(pay[0, 0] - 20.0) < 1e-9                                            # 2 * c2 * d
    assert abs(pay[0, 0] - 1.2 * 10.0 - 8.0) < 1e-9                                # Nutzen 8 = 10 * (2.0 - 1.2)
    steps = work_steps(world.S, c[0], 0, 0.05, 4.0)
    assert [round(end, 6) for _, end, _ in steps[:-1]] == [0.5, 2.0]               # Sprungstellen c2/2 und 2*c2
    assert [v for _, _, v in steps] == [20.0, 10.0, 0.0]


def test_untruncated_threshold_integral_would_overpay_on_the_vehicle_instance():
    world = WorldB(4, 2)
    c = np.array([1.0, 1.0])
    beyond = work_curve(world.S, c, 0, np.array([C_HI + 1e-6, 5.0]))
    assert beyond.max() > 0            # ueber C_HI ist die Arbeit noch positiv: ein Integral bis unendlich wuerde weiterlaufen und ueberzahlen


def test_threshold_is_truthful_and_individually_rational_on_a_sample():
    world = WorldB(3, 2)
    c = world.heldout(6)
    assert threshold_regret_sample(world, c, grid_points=25) < 1e-6
    pi, pay = threshold_mechanism(world, c)
    assert (pay - np.einsum("np,npk->nk", pi, world.costs(c))).min() > -1e-9


def test_work_curve_is_non_increasing():
    world = WorldB(3, 3)
    for row in world.heldout(5):
        for a in range(3):
            w = work_curve(world.S, row, a, np.linspace(C_LO, 4.0, 300))
            assert (np.diff(w) <= 1e-9).all()


# --- Netz, Verlust, Regret ---------------------------------------------------------------------------------

@pytest.mark.parametrize("world", [WorldB(3, 2), WorldA(3, 2)], ids=["B", "A"])
def test_surrogate_gradient_matches_finite_differences(world):
    rng = np.random.default_rng(0)
    net = make_net(world, 8, rng)
    for w in net.W:
        w += 0.3 * rng.standard_normal(w.shape)
    reports = world.sample(rng, 16)
    found = []
    for a in range(world.k):
        if world.kind == "B":
            rep = world.replace(reports, a, rng.uniform(0.5, 2, 16))
        else:
            rep = reports.copy()
            rep[:, a, 1:] *= rng.uniform(0.6, 2, (16, world.total - 1))
        found.append((rep, rng.random(16) < 0.7))
    kappa = np.array([3.0, 5.0])
    _, grads = surrogate(world, net, reports, found, kappa, 0.1)
    worst = 0.0
    for gi, arr in enumerate(net.params()):
        for _ in range(4):
            ix = tuple(rng.integers(0, s) for s in arr.shape)
            old = arr[ix]
            arr[ix] = old + 1e-6
            lp, _ = surrogate(world, net, reports, found, kappa, 0.1)
            arr[ix] = old - 1e-6
            lm, _ = surrogate(world, net, reports, found, kappa, 0.1)
            arr[ix] = old
            worst = max(worst, abs((lp - lm) / 2e-6 - grads[gi][ix]))
    assert worst < 1e-6


def test_input_gradient_of_the_network_matches_finite_differences():
    rng = np.random.default_rng(1)
    world = WorldA(3, 2)
    net = make_net(world, 8, rng)
    for w in net.W:
        w += 0.3 * rng.standard_normal(w.shape)
    x = rng.standard_normal((5, world.dim))
    weights = rng.standard_normal((5, world.P + world.k))
    hs = net.forward(x)
    _, dx = net.backward(hs, weights, want_input_grad=True)
    for j in range(world.dim):
        xp, xm = x.copy(), x.copy()
        xp[:, j] += 1e-6
        xm[:, j] -= 1e-6
        fd = ((net.forward(xp)[-1] * weights).sum(1) - (net.forward(xm)[-1] * weights).sum(1)) / 2e-6
        assert np.allclose(fd, dx[:, j], atol=1e-6)


def test_learned_mechanism_probabilities_and_ir_by_construction():
    rng = np.random.default_rng(0)
    world = WorldB(3, 2)
    lm = LearnedMechanism(world, make_net(world, 8, rng))
    c = world.sample(rng, 20)
    pi, pay = lm(world, c)
    assert np.allclose(pi.sum(1), 1.0) and (pi >= 0).all()
    assert (pay - np.einsum("np,npk->nk", pi, world.costs(c))).min() > 0            # Nutzen = Rente > 0
    rounded_pi, _ = lm.rounded_copy()(world, c)
    assert set(np.unique(rounded_pi)) <= {0.0, 1.0}


def test_regret_of_pay_as_bid_in_type_b_is_large_and_of_threshold_is_zero_on_sample():
    world = WorldB(3, 2)
    c = world.heldout(20)
    assert regret_type_b(world, hand("A"), c, 60)[0].mean() > 5
    assert regret_type_b(world, lambda w, r: threshold_mechanism(w, r), c[:6], 20)[0].max() < 1e-6


def test_training_is_deterministic_and_improves_over_untrained():
    world = WorldB(2, 2)
    cfg = TrainConfig(hidden=16, batch=64, steps=300, warmup=50, seed=3)
    a, b = train(world, cfg), train(world, cfg)
    assert all(np.array_equal(x, y) for x, y in zip(a.mechanism.net.params(), b.mechanism.net.params()))
    c = world.heldout(60)
    untrained = LearnedMechanism(world, make_net(world, 16, np.random.default_rng(3)))
    costs = world.costs(c)

    def gap(m):
        pi, _ = m(world, c)
        return (pi * costs.max(-1)).sum(1).mean() / costs.max(-1).min(1).mean() - 1

    assert gap(a.mechanism) < gap(untrained)


def test_quick_live_training_reaches_the_documented_quality_on_the_hand_example():
    world = WorldB(2, 2)
    result = train(world, replace(QUICK_B, steps=1000))
    rows = {r["key"]: r for r in comparison_table(world, world.heldout(200), result.mechanism)}
    assert rows["learned"]["gap_pct"] < 3.0 and rows["learned"]["regret"] < 0.2
    assert rows["threshold"]["gap_pct"] < 1e-9 and rows["threshold"]["regret"] == 0.0
    assert result.seconds < 30


# --- Speichern/Laden, Test-Seeds, Konvention ------------------------------------------------------------

def test_save_and_load_roundtrip(tmp_path):
    world = WorldB(3, 2)
    lm = LearnedMechanism(world, make_net(world, 8, np.random.default_rng(0)))
    save_mechanism(tmp_path / "m.npz", world, lm.net, {"name": "t"})
    loaded, w2, meta = load_mechanism(tmp_path / "m.npz")
    c = world.heldout(10)
    assert np.allclose(lm(world, c)[0], loaded(w2, c)[0]) and meta["name"] == "t" and (w2.n, w2.k) == (3, 2)


def test_precomputed_weights_load_and_stay_in_documented_bands():
    from rn_evaluation import basic_metrics
    lm, world, _ = load_mechanism(ROOT / "weights" / "b_n4k2_k3.npz")
    metrics = basic_metrics(world, lm, world.heldout(C.N_TEST_B))
    assert 0.1 < metrics["gap_pct"] < 3.5 and 1.5 < metrics["pay_cost"] < 1.62
    for name in C.WEIGHT_SPECS:
        assert (ROOT / "weights" / f"{name}.npz").exists()


def test_held_out_seeds_never_collide_with_demo_or_training_seeds():
    assert HELDOUT_SEED_BASE >= 100_000 > C.DEFAULT_SEED
    a, b = WorldB(4, 2).heldout(50), WorldB(4, 2).heldout(50)
    assert np.array_equal(a, b) and a.min() >= C_LO and a.max() <= C_HI


def test_every_figure_of_the_visualisation_module_is_axis_locked():
    source = (ROOT / "rn_visualization.py").read_text(encoding="utf-8")
    assert not re.search(r"return fig\b", source)
    assert source.count("return lock_axes(fig)") >= 6
