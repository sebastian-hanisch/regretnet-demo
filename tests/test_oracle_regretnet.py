"""Unabhängige Orakel (reine Aufzählung, eigene Schleifen): Bündelgebote gegen Permutations-Brute-Force, Handregeln A/B/E/Boost
gegen Vollenumeration aller Zuteilungen mit Clarke-Zahlung per Definition, Schwellenwert-Zahlung gegen Riemann-Summe der
Arbeitskurve, Regret (Typ B) gegen eine Schleife über Fehlmeldungen und Handbeispiel."""
import itertools
import math
import random

import numpy as np

from auction_bids import bundle_bid_table
from cn_scenario import generate_instance
from rn_baselines import hand_mechanism, threshold_mechanism
from rn_regret import regret_type_b
from rn_types import C_HI, C_LO, WorldA, WorldB


def _bid_bruteforce(inst, a, jobs):
    if not jobs:
        return 0.0
    best = math.inf
    for perm in itertools.permutations(jobs):
        pos, t = inst.agent_start_positions[a], 0.0
        for j in perm:
            t += abs(pos - inst.jobs[j].position) * inst.travel_time_per_unit + inst.jobs[j].duration
            pos = inst.jobs[j].position
        best = min(best, t)
    return best


def test_bundle_bids_match_permutation_bruteforce():
    rng = random.Random(1)
    for _ in range(25):
        n, k = rng.randint(1, 4), rng.randint(1, 3)
        inst = generate_instance(n, k, rng.choice([0, 0.5]), rng.choice([0.2, 1.0, 2.0]), rng.randrange(10 ** 5))
        table = bundle_bid_table(inst)
        for a in range(k):
            for m in range(1 << n):
                jobs = tuple(j for j in range(n) if (m >> j) & 1)
                assert abs(table[a, m] - _bid_bruteforce(inst, a, jobs)) < 1e-9


def _ref_rule(W, n, k, rule, gamma):
    rows = []
    for owner in itertools.product(range(k), repeat=n):
        ms = tuple(sum(1 << j for j in range(n) if owner[j] == a) for a in range(k))
        cs = [W[a][m] for a, m in enumerate(ms)]
        bonus = gamma * sum(bin(m).count("1") ** 2 for m in ms) if rule == "boost" else 0.0
        rows.append((ms, cs, sum(cs), max(cs), bonus))

    def pick(rows_):
        first = (lambda r: r[3]) if rule in ("A", "E") else (lambda r: r[2] + r[4])
        second = (lambda r: r[2]) if rule in ("A", "E") else (lambda r: r[3])
        m0 = min(first(r) for r in rows_)
        cand = [r for r in rows_ if first(r) <= m0 + 1e-9 * max(1, abs(m0))]
        m1 = min(second(r) for r in cand)
        return [r for r in cand if second(r) <= m1 + 1e-9 * max(1, abs(m1))][0]

    ms, cs, total, mk, bonus = pick(rows)
    pays = []
    for a in range(k):
        if ms[a] == 0:
            pays.append(0.0)
        elif rule == "A":
            pays.append(cs[a])
        else:
            without = min(r[2] + r[4] for r in rows if r[0][a] == 0)
            pays.append(without - (total + bonus - cs[a]))
    return ms, pays, (mk, total)


def test_hand_rules_match_full_enumeration_with_clarke_by_definition():
    rng = np.random.default_rng(4)
    for kind, n, k in [("B", 3, 2), ("B", 3, 3), ("A", 3, 2), ("B", 4, 2)]:
        world = WorldB(n, k, instance_seed=int(rng.integers(1000))) if kind == "B" else WorldA(n, k)
        reports = world.sample(rng, 6)
        tab = bundle_bid_table(world.instance) if kind == "B" else None
        for rule, gamma in [("A", 0.0), ("B", 0.0), ("E", 0.0), ("boost", 3.0)]:
            pi, pay = hand_mechanism(world, reports, rule, gamma)
            for i, r in enumerate(reports):
                if kind == "B":
                    W = [[r[a] * tab[a, m] for m in range(1 << n)] for a in range(k)]
                else:
                    W = [[r[a, m] for m in range(1 << n)] for a in range(k)]
                ms, pays, key = _ref_rule(W, n, k, rule, gamma)
                mine = tuple(int(m) for m in world.masks[int(pi[i].argmax())])
                cs = [W[a][m] for a, m in enumerate(mine)]
                assert abs(max(cs) - key[0]) < 1e-9 and abs(sum(cs) - key[1]) < 1e-9
                if mine == ms:
                    np.testing.assert_allclose(pay[i], pays, atol=1e-7)


def _allocate(S, c):
    best_key, best_p = None, None
    for p in range(S.shape[0]):
        cs = c * S[p]
        key = (cs.max(), cs.sum())
        tol = 1e-9 * max(1, abs(best_key[0])) if best_key else 0.0
        if best_key is None or key[0] < best_key[0] - tol or (abs(key[0] - best_key[0]) <= tol and key[1] < best_key[1] - 1e-12):
            best_key, best_p = key, p
    return best_p


def test_threshold_payment_matches_riemann_sum_of_the_work_curve():
    rng = np.random.default_rng(2)
    for n, k in [(2, 2), (3, 2), (3, 3)]:
        world = WorldB(n, k, instance_seed=int(rng.integers(1000)))
        c = world.sample(rng, 3)
        _, pay = threshold_mechanism(world, c)
        M = 1500
        for i in range(len(c)):
            for a in range(k):
                b = c[i, a]
                us = b + (C_HI - b) * (np.arange(M) + 0.5) / M
                ws = []
                for u in us:
                    cc = c[i].copy()
                    cc[a] = u
                    ws.append(world.S[_allocate(world.S, cc), a])
                ref = b * world.S[_allocate(world.S, c[i]), a] + float(np.sum(ws) * (C_HI - b) / M)
                assert abs(ref - pay[i, a]) < 0.02 * max(1.0, ref)


def test_hand_example_payment_20_and_utility_8():
    world = WorldB(2, 2)
    c = np.array([[1.2, 1.0]])
    pi, pay = threshold_mechanism(world, c)
    assert abs(pay[0, 0] - 20.0) < 1e-9
    assert abs(pay[0, 0] - 1.2 * world.S[int(pi[0].argmax()), 0] - 8.0) < 1e-9


def test_regret_type_b_matches_loop_over_misreports():
    world = WorldB(3, 2, instance_seed=3)
    reports = world.heldout(5)
    def mech(w, r):
        return hand_mechanism(w, r, "E")

    tab = bundle_bid_table(world.instance)
    grid = np.linspace(C_LO, C_HI, 200)
    got, _ = regret_type_b(world, mech, reports, 200)
    for i, r in enumerate(reports):
        for a in range(2):
            def util(rep):
                pi, pay = mech(world, rep[None, :])
                return pay[0, a] - sum(pi[0][p] * r[a] * tab[a, world.masks[p, a]] for p in range(world.P))
            u0 = util(r)
            best = max([u0] + [util(np.where(np.arange(2) == a, g, r)) for g in grid])
            assert abs(max(best - u0, 0.0) - got[i, a]) < 1e-9
