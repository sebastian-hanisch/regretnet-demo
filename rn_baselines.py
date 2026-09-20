"""Handentworfene Mechanismen als Vergleichsschicht, vektorisiert über Stichproben auf `costs(reports) -> [N, P, k]`.

Jeder Mechanismus ist ein Aufruf `mech(world, reports) -> (pi [N, P], pay [N, k])`: `pi` ist die Zuteilungs-Wahrscheinlichkeit über die
k^n Zuteilungen (bei deterministischen Regeln one-hot), `pay` die Zahlung je Agent. Nutzen des Agenten a bei wahren Kosten C_true:
u_a = pay_a − Σ_p pi_p · C_true[p, a].

Regeln (Ziel der Zuteilung × Zahlung) - dieselben wie in auction-demo, dort gegen `auction_mechanism.run_mechanism` getestet:
  "A"   Makespan-Zuteilung + Pay-as-bid
  "B"   Summen-Zuteilung + VCG (Clarke)
  "E"   Makespan-Zuteilung + Clarke-Formel (Zahlung kann unter die Kosten fallen)
  "boost" (gamma)  Summe + gamma · Σ_a |Bündel_a|² + gewichtete Clarke-Zahlung - EIGENE Konstruktion (kein Zitat): ein
        gebotsunabhängiger Bonus auf gleichmäßige Bündelgrößen; die Zuteilung ist ein affiner Maximierer, die Zahlung nach Clarke mit
        dem Bonus im Zielwert - exakt wahrheitsgetreu und individuell rational. gamma = 0 ist VCG.
Nur für Typ B (ein Parameter): `threshold_mechanism` = Makespan-Zuteilung (Tie-Break Gesamtkosten) + gestutzte Schwellenwert-Zahlung
(Archer & Tardos 2001: Wahrhaftigkeit ⇔ monotone Arbeitskurve, Zahlung b·w(b) + ∫_b^∞ w; das Abschneiden des Integrals am Typbereich
C_HI ist EIGENE, numerisch geprüfte Anpassung - ohne sie überzahlt die Formel auf dem beschränkten Typbereich um ein Mehrfaches)."""

import numpy as np

from rn_types import C_HI

TOL = 1e-9


def _argmin_lex(score, tie):
    """Erstes Minimum von `score` je Zeile, Gleichstand (Toleranz) nach `tie`, dann nach kleinstem Index. [N, P] -> [N]."""
    best = score.min(1, keepdims=True)
    candidates = score <= best + TOL * np.maximum(1.0, np.abs(best))
    return np.where(candidates, tie, np.inf).argmin(1)


def _one_hot(index, P):
    pi = np.zeros((len(index), P))
    pi[np.arange(len(index)), index] = 1.0
    return pi


def hand_mechanism(world, reports, rule, gamma=0.0):
    C = world.costs(reports)                                   # [N, P, k]
    N, P, k = C.shape
    rows = np.arange(N)
    total, makespan = C.sum(-1), C.max(-1)
    empty = (world.masks == 0)                                 # [P, k]: Agent a hat in Zuteilung p nichts zu tun
    if rule == "A":
        p = _argmin_lex(makespan, total)
        pay = C[rows, p, :]
    else:
        if rule in ("B", "boost"):
            objective = total + (gamma * (world.sizes ** 2).sum(-1))[None, :] if rule == "boost" else total
            p = _argmin_lex(objective, makespan)
        elif rule == "E":
            objective = total
            p = _argmin_lex(makespan, total)
        else:
            raise ValueError(rule)
        pay = np.zeros((N, k))
        chosen_cost = C[rows, p, :]                            # [N, k]
        chosen_obj = objective[rows, p]                        # [N]
        for a in range(k):
            without = np.where(empty[None, :, a], objective, np.inf).min(1)     # Zielwert, wenn Agent a nicht teilnimmt
            pay[:, a] = without - (chosen_obj - chosen_cost[:, a])
    pay = np.where(world.masks[p] == 0, 0.0, pay)                              # Verlierer bekommen nichts
    return _one_hot(p, P), pay


def threshold_mechanism(world, c):
    """Exakte gestutzte Schwellenwert-Zahlung (nur Typ B). c [N, k] -> (pi [N, P] one-hot, pay [N, k])."""
    N, k, P = len(c), world.k, world.P
    S = world.S                                                # [P, k] Bündelzeit
    C = c[:, None, :] * S[None, :, :]
    p_star = _argmin_lex(C.max(-1), C.sum(-1))
    pay = np.zeros((N, k))
    for n in range(N):
        for a in range(k):
            pay[n, a] = _threshold_payment(S, c[n], a, p_star[n])
    return _one_hot(p_star, P), pay


def work_curve(S, c_row, a, u):
    """Arbeit w(u) von Agent a (Bündelzeit seiner Zuteilung), wenn er u meldet und die anderen c_row melden - Makespan-Zuteilung mit
    Tie-Break Gesamtkosten. u [M] -> [M]."""
    others = np.delete(np.arange(len(c_row)), a)
    other_cost = c_row[others][None, :] * S[:, others]                        # [P, k-1]
    R = other_cost.max(1) if len(others) else np.zeros(len(S))                # Makespan der anderen je Zuteilung
    T = other_cost.sum(1)
    s = S[:, a]
    F = np.maximum(u[:, None] * s[None, :], R[None, :])                       # [M, P]
    tie = u[:, None] * s[None, :] + T[None, :]
    best = F.min(1, keepdims=True)
    cand = F <= best + TOL * np.maximum(1.0, np.abs(best))
    idx = np.where(cand, tie, np.inf).argmin(1)
    return s[idx]


def breakpoints(S, c_row, a, lo, hi):
    """Alle Stellen in (lo, hi), an denen sich die Zuteilung von Agent a ändern kann (Knicke, Kreuzungen, Tie-Break-Wechsel)."""
    others = np.delete(np.arange(len(c_row)), a)
    other_cost = c_row[others][None, :] * S[:, others]
    R = other_cost.max(1) if len(others) else np.zeros(len(S))
    T = other_cost.sum(1)
    s = S[:, a]
    pts = []
    pos = s > 0
    if pos.any():
        pts.append((R[None, :] / s[pos][:, None]).ravel())                   # u * s_p = R_q
    ds = s[:, None] - s[None, :]
    dT = T[None, :] - T[:, None]
    ok = np.abs(ds) > 1e-12
    pts.append((dT[ok] / ds[ok]))                                             # Tie-Break-Gleichstand u s_p + T_p = u s_q + T_q
    pts = np.concatenate(pts)
    pts = pts[(pts > lo + 1e-12) & (pts < hi - 1e-12)]
    return np.unique(np.round(pts, 12))


def work_steps(S, c_row, a, lo, hi):
    """Die Stufen der Arbeitskurve w(u) von Agent a auf [lo, hi]: Liste (von, bis, Arbeit), gleiche Nachbarn zusammengefasst."""
    cuts = np.concatenate([[lo], breakpoints(S, c_row, a, lo, hi), [hi]])
    mids = 0.5 * (cuts[:-1] + cuts[1:])
    w = work_curve(S, c_row, a, mids)
    steps = []
    for start, end, value in zip(cuts[:-1], cuts[1:], w):
        if steps and abs(steps[-1][2] - value) < 1e-9:
            steps[-1] = (steps[-1][0], float(end), float(value))
        else:
            steps.append((float(start), float(end), float(value)))
    return steps


def _threshold_payment(S, c_row, a, p_truth):
    """P_a(b) = b·w(b) + ∫_b^{C_HI} w(u) du mit b = c_row[a] (exakt: w ist eine Stufenfunktion)."""
    b = c_row[a]
    w_b = S[p_truth, a]
    cuts = np.concatenate([[b], breakpoints(S, c_row, a, b, C_HI), [C_HI]])
    mids = 0.5 * (cuts[:-1] + cuts[1:])
    w = work_curve(S, c_row, a, mids)
    return b * w_b + float((w * np.diff(cuts)).sum())


def utilities(world, mech, reports, true_costs=None):
    """Nutzen aller Agenten bei ehrlichen Meldungen = Zahlung − erwartete wahre Kosten. -> [N, k]"""
    pi, pay = mech(world, reports)
    C = world.costs(reports) if true_costs is None else true_costs
    return pay - np.einsum("np,npk->nk", pi, C)
