"""Alle Zuteilungen von n Aufträgen an k Agenten ("Partitionen") als Tabellen - die Ausgabeköpfe der Netze sind Softmax über
diese k^n Zuteilungen, damit XOR (ein Bündel je Agent) und Zulässigkeit (jeder Auftrag genau einmal) per Konstruktion gelten.

Zuteilung p: Auftrag j gehört Agent (p // k^j) % k - dieselbe Kodierung wie `auction_wd.wd_bruteforce`. `masks[p, a]` ist das
Bündel (Bitmaske) von Agent a in Zuteilung p, 0 = leeres Bündel."""

import numpy as np

_CACHE = {}


def partitions(n_jobs, n_agents):
    """(masks [P, k] int, sizes [P, k] int) mit P = k^n."""
    key = (n_jobs, n_agents)
    if key not in _CACHE:
        k, n = n_agents, n_jobs
        P = k ** n
        codes = np.arange(P)
        masks = np.zeros((P, k), dtype=np.int64)
        for j in range(n):
            owner = (codes // k ** j) % k
            masks[codes, owner] |= 1 << j
        sizes = np.zeros((P, k), dtype=np.int64)
        for j in range(n):
            sizes += (masks >> j) & 1
        _CACHE[key] = (masks, sizes)
    return _CACHE[key]


def gather_costs(bids, masks):
    """bids [N, k, 2^n] (Spalte 0 = 0) -> Kosten je Zuteilung [N, P, k]: C[n, p, a] = bids[n, a, masks[p, a]]."""
    k = bids.shape[1]
    return bids[:, np.arange(k)[None, :], masks]


def gather_costs_single(table, masks):
    """table [k, 2^n] -> [P, k] (eine feste Instanz)."""
    k = table.shape[0]
    return table[np.arange(k)[None, :], masks]
