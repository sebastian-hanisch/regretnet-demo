"""Gewinnerermittlung (Winner Determination, WD): welche Bündel gewinnen? Bei XOR-Geboten (ein Bündel je Agent) ist das eine
MIN-MAX-MENGENPARTITION - jeder Auftrag genau einmal, jeder Agent höchstens ein Bündel, minimiert wird das Maximum der
Zuschlags-Gebote (= der Makespan). Das Summenziel (`objective="sum"`) minimiert stattdessen die Gesamtzeit.

Exakt über einen Teilmengen-DP (3^n Teilmengen-Paare je Agent): F_a[T] = min über S ⊆ T von max(F_{a-1}[T \\ S], bid_a[S]).
NP-hart im Allgemeinen - die Zeit wächst mit 3^n, die Gebotstabelle mit 2^n; die Mauer liegt bei n ~ 16.

`blocks_wd_poly` ist der polynomielle Sonderfall der Ein-Block-Sprache (zusammenhängende Blöcke im nach Position sortierten
Auftragsstrom): DP über Präfix x Agentenmenge."""

import time
from dataclasses import dataclass

import numpy as np

from auction_bids import INF, jobs_of

_ACTIVE_CACHE = {}


@dataclass(frozen=True)
class WDResult:
    value: float          # Makespan (objective="max") bzw. Gesamtzeit ("sum"); ∞ wenn nicht zulässig
    masks: tuple          # Bündel-Maske je Agent (0 = leer)
    feasible: bool
    seconds: float
    pairs: int            # ausgewertete Teilmengen-Paare = k * 3^n


def _subset_plan(n):
    """Reihenfolge und Zähler, damit ALLE Teilmengen jeder Maske vektorisiert durchlaufen werden können (numpy)."""
    if n in _ACTIVE_CACHE:
        return _ACTIVE_CACHE[n]
    total = 1 << n
    pop = np.zeros(total, dtype=np.int64)
    for j in range(n):
        pop[1 << j:2 << j] = pop[:1 << j] + 1
    order = np.argsort(-pop, kind="stable")           # Popcount absteigend
    iterations = 1 << pop[order]                       # 2^popcount Teilmengen je Maske
    steps = int(iterations[0])
    counts = np.searchsorted(-iterations, -np.arange(steps), side="left")
    _ACTIVE_CACHE[n] = (order.astype(np.int64), counts)
    return _ACTIVE_CACHE[n]


def winner_determination(bids, objective="max"):
    """Exakte WD. bids: [k, 2^n] (∞ = nicht angeboten, Spalte 0 = leeres Bündel mit 0). Gibt ein WDResult zurück."""
    started = time.perf_counter()
    k, total = bids.shape
    n = total.bit_length() - 1
    order, counts = _subset_plan(n)
    prev = np.full(total, INF)
    prev[0] = 0.0
    choices = []
    for a in range(k):
        subset = order.copy()
        best = np.full(total, INF)
        best_subset = np.zeros(total, dtype=np.int64)
        for step in range(len(counts)):
            count = counts[step]
            if count == 0:
                break
            current = subset[:count]
            remaining = order[:count] ^ current
            if objective == "max":
                value = np.maximum(prev[remaining], bids[a][current])
            else:
                value = prev[remaining] + bids[a][current]
            better = value < best[:count]
            best[:count] = np.where(better, value, best[:count])
            best_subset[:count] = np.where(better, current, best_subset[:count])
            subset[:count] = (current - 1) & order[:count]
        new = np.full(total, INF)
        chosen = np.zeros(total, dtype=np.int64)
        new[order] = best
        chosen[order] = best_subset
        choices.append(chosen)
        prev = new
    full = total - 1
    value = float(prev[full])
    feasible = bool(np.isfinite(value))
    masks = [0] * k
    if feasible:
        mask = full
        for a in range(k - 1, -1, -1):
            masks[a] = int(choices[a][mask])
            mask ^= masks[a]
    return WDResult(value, tuple(masks), feasible, time.perf_counter() - started, k * 3 ** n)


def wd_bruteforce(bids, objective="max"):
    """Orakel für Tests (n <= 6): alle k^n Zuteilungen von Aufträgen zu Agenten aufzählen."""
    k, total = bids.shape
    n = total.bit_length() - 1
    best = INF
    for code in range(k ** n):
        masks = [0] * k
        x = code
        for j in range(n):
            masks[x % k] |= 1 << j
            x //= k
        values = [bids[a][masks[a]] for a in range(k)]
        value = max(values) if objective == "max" else sum(values)
        best = min(best, value)
    return float(best)


def wd_cpsat(bids, time_limit_seconds=5.0):
    """CP-SAT als Mengenpartitions-Löser auf der Gebotstabelle (Vergleichsknopf, Tests): Bool je (Agent, Bündel), jeder Auftrag
    genau einmal, je Agent höchstens ein Bündel, minimiert wird der Makespan. Gibt (Wert, bewiesen optimal, Sekunden)."""
    from ortools.sat.python import cp_model

    started = time.perf_counter()
    k, total = bids.shape
    n = total.bit_length() - 1
    scale = 1000
    model = cp_model.CpModel()
    variables = {}
    for a in range(k):
        for mask in range(1, total):
            if np.isfinite(bids[a][mask]):
                variables[(a, mask)] = model.NewBoolVar(f"x_{a}_{mask}")
    for a in range(k):
        model.Add(sum(v for (aa, _), v in variables.items() if aa == a) <= 1)
    for j in range(n):
        model.Add(sum(v for (_, mask), v in variables.items() if (mask >> j) & 1) == 1)
    makespan = model.NewIntVar(0, int(np.nanmax(bids[np.isfinite(bids)]) * scale) + 1, "makespan")
    for (a, mask), v in variables.items():
        model.Add(makespan >= int(np.ceil(bids[a][mask] * scale - 1e-9))).OnlyEnforceIf(v)
    model.Minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_workers = 1
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return INF, False, time.perf_counter() - started
    return solver.ObjectiveValue() / scale, status == cp_model.OPTIMAL, time.perf_counter() - started


def blocks_wd_poly(instance):
    """Exakte WD der Ein-Block-Sprache in polynomieller Zeit: der nach Position sortierte Auftragsstrom wird in höchstens k
    zusammenhängende Blöcke zerlegt, je einer pro Agent. DP über (Präfix i, benutzte Agentenmenge U). Gibt (Wert, Masken)."""
    n, k = instance.n_jobs, instance.n_agents
    tau = instance.travel_time_per_unit
    order = sorted(range(n), key=lambda j: (instance.jobs[j].position, j))
    pos = np.array([instance.jobs[j].position for j in order])
    dur = np.array([instance.jobs[j].duration for j in order])
    cumulative = np.concatenate([[0.0], np.cumsum(dur)])

    def bid(agent, i, j):                      # Block der sortierten Aufträge i..j-1
        lo, hi = pos[i], pos[j - 1]
        start = instance.agent_start_positions[agent]
        return cumulative[j] - cumulative[i] + tau * ((hi - lo) + min(abs(start - lo), abs(start - hi)))

    subsets = 1 << k
    value = np.full((n + 1, subsets), INF)
    value[0, 0] = 0.0
    back = {}
    for i in range(1, n + 1):
        for used in range(1, subsets):
            best, best_choice = INF, None
            for agent in range(k):
                if not (used >> agent) & 1:
                    continue
                rest = used ^ (1 << agent)
                for j in range(i):
                    if value[j, rest] < INF:
                        candidate = max(value[j, rest], bid(agent, j, i))
                        if candidate < best:
                            best, best_choice = candidate, (agent, j, rest)
            value[i, used] = best
            back[(i, used)] = best_choice
    used = int(np.argmin(value[n, 1:])) + 1
    result = float(value[n, used])
    masks = [0] * k
    i = n
    while i > 0:
        agent, j, rest = back[(i, used)]
        for idx in range(j, i):
            masks[agent] |= 1 << order[idx]
        i, used = j, rest
    return result, tuple(masks)


__all__ = ["WDResult", "winner_determination", "wd_bruteforce", "wd_cpsat", "blocks_wd_poly", "jobs_of"]
