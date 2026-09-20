"""Gebote auf Auftragsbündel. Das Gebot eines Agenten auf ein Bündel S ist die Fertigstellungszeit seiner besten Route
durch S ab seiner Startposition ("wann bin ich fertig, wenn ich genau diese Aufträge übernehme?"). In diesem 1-D-Vehikel
ist die beste Route geschlossen bekannt:

    bid(S) = Σ dauer + τ · ((hi − lo) + min(|s − lo|, |s − hi|))

(erst zum näheren Ende der Positions-Spanne fahren, dann in einem Zug bis zum anderen Ende - Sweep). Das ist gegen eine
Held-Karp-Teilmengen-DP geprüft. Ein 2-D-Vehikel hätte hier ein NP-hartes Weg-Problem (TSP-Pfad); dort wäre schon die
Gebotsberechnung teuer - der 2-D-Transfer der Ergebnisse ist vermutet, nicht gemessen.

Bündel sind Bitmasken über die Aufträge (Bit j = Auftrag j); die leere Menge hat Gebot 0 (der Agent bekommt nichts).
Ein Agent gewinnt HÖCHSTENS EIN Bündel (XOR-Gebote): dann ist der Makespan exakt das Maximum der Zuschlags-Gebote."""

from math import comb

import numpy as np

import cn_constants as C

INF = float("inf")


def bundle_bid_table(instance):
    """Gebote ALLER Agenten auf ALLE Bündel: Array [k, 2^n], Spalte 0 (leere Menge) ist 0."""
    n, k = instance.n_jobs, instance.n_agents
    total = 1 << n
    pos = np.array([j.position for j in instance.jobs])
    dur = np.array([j.duration for j in instance.jobs])
    lo = np.full(total, INF)
    hi = np.full(total, -INF)
    dur_sum = np.zeros(total)
    for j in range(n):
        step = 1 << j
        lo[step:2 * step] = np.minimum(lo[:step], pos[j])
        hi[step:2 * step] = np.maximum(hi[:step], pos[j])
        dur_sum[step:2 * step] = dur_sum[:step] + dur[j]
    lo[0] = hi[0] = 0.0
    table = np.zeros((k, total))
    for a in range(k):
        start = instance.agent_start_positions[a]
        table[a] = dur_sum + instance.travel_time_per_unit * (
            (hi - lo) + np.minimum(np.abs(start - lo), np.abs(start - hi))
        )
    table[:, 0] = 0.0
    return table


def jobs_of(mask, n_jobs):
    return tuple(j for j in range(n_jobs) if (mask >> j) & 1)


def mask_of(jobs):
    mask = 0
    for j in jobs:
        mask |= 1 << j
    return mask


def bundle_route(instance, agent_id, mask):
    """Besuchsreihenfolge des Agenten durch das Bündel: Sweep vom näheren Ende der Positions-Spanne (bei Gleichstand
    aufsteigend). Gleiche Positionen behalten aufsteigende Auftrags-Reihenfolge."""
    jobs = jobs_of(mask, instance.n_jobs)
    if not jobs:
        return ()
    positions = [instance.jobs[j].position for j in jobs]
    lo, hi = min(positions), max(positions)
    start = instance.agent_start_positions[agent_id]
    if abs(start - lo) <= abs(start - hi):
        return tuple(sorted(jobs, key=lambda j: (instance.jobs[j].position, j)))
    return tuple(sorted(jobs, key=lambda j: (-instance.jobs[j].position, j)))


def bundle_bid(instance, agent_id, jobs):
    """Gebot eines Agenten auf ein einzelnes Bündel (Tupel von Auftragsindizes)."""
    if not jobs:
        return 0.0
    positions = [instance.jobs[j].position for j in jobs]
    lo, hi = min(positions), max(positions)
    start = instance.agent_start_positions[agent_id]
    return sum(instance.jobs[j].duration for j in jobs) + instance.travel_time_per_unit * (
        (hi - lo) + min(abs(start - lo), abs(start - hi))
    )


# --- Gebotssprachen -----------------------------------------------------------------------------

def _popcount(n):
    counts = np.zeros(1 << n, dtype=np.int16)
    for j in range(n):
        counts[1 << j:2 << j] = counts[:1 << j] + 1
    return counts


def _interval_masks(instance):
    n = instance.n_jobs
    order = sorted(range(n), key=lambda j: (instance.jobs[j].position, j))
    intervals = []
    for lo in range(n):
        mask = 0
        for hi in range(lo, n):
            mask |= 1 << order[hi]
            intervals.append(mask)
    return intervals


def language_mask(instance, language, b=None):
    """Bool-Array [2^n]: welche Bündel bieten die Agenten an? (Maske 0 immer True.) Blöcke = zusammenhängende
    Abschnitte des nach Position sortierten Auftragsstroms."""
    n = instance.n_jobs
    total = 1 << n
    if language == C.LANG_ALL:
        return np.ones(total, dtype=bool)
    if language == C.LANG_SIZE:
        return _popcount(n) <= int(b)
    mask = np.zeros(total, dtype=bool)
    mask[0] = True
    intervals = _interval_masks(instance)
    for m in intervals:
        mask[m] = True
    if language == C.LANG_BLOCK2:
        for first in intervals:
            for second in intervals:
                if first & second == 0:
                    mask[first | second] = True
        return mask
    if language == C.LANG_BLOCK1:
        return mask
    raise ValueError(f"unbekannte Sprache: {language}")


def apply_language(bids, mask):
    """Nicht angebotene Bündel bekommen Gebot ∞."""
    return np.where(mask[None, :], bids, INF)


def min_bundle_size(n_jobs, n_agents):
    """Kleinste Bündelgröße, mit der bei XOR-Geboten (ein Bündel je Agent) überhaupt alle Aufträge vergeben werden können."""
    return -(-n_jobs // n_agents)


def n_bids(n_jobs, n_agents, language, b=None):
    """Anzahl Gebote (nichtleere Bündel je Agent x Agenten) als geschlossene Formel."""
    n, k = n_jobs, n_agents
    if language == C.LANG_ALL:
        return k * ((1 << n) - 1)
    if language == C.LANG_SIZE:
        return k * sum(comb(n, s) for s in range(1, min(int(b), n) + 1))
    if language == C.LANG_BLOCK1:
        return k * n * (n + 1) // 2
    if language == C.LANG_BLOCK2:
        return k * sum(comb(n, i) for i in range(1, min(4, n) + 1))
    raise ValueError(f"unbekannte Sprache: {language}")


def n_bids_cnp(n_jobs, n_agents):
    return n_jobs * n_agents
