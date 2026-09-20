"""Auswertung: alle Mechanismen (Handregeln, exakte Schwelle, gelernt, gelernt+gerundet) auf festen Test-Stichproben.

Kennzahlen je Mechanismus (Test-Seeds ab 100000, unabhängig vom Trainings- und Demo-Seed):
  gap_pct   erwarteter Makespan über dem Optimum in % (gepoolt: Mittel des erwarteten Makespans / Mittel des Optimums − 1)
  pay_cost  Zahlung / erwartete wahre Kosten (gepoolt) - die Überzahlung
  ir_share  Anteil der (Instanz, Agent)-Paare mit negativem Nutzen bei ehrlichem Bieten (ex-interim-IR verletzt)
  regret    mittlerer Regret je Agent in Minuten (Untergrenze!) und Anteil der Agenten mit Regret > 0.1 min"""

import numpy as np

from rn_baselines import _argmin_lex, _threshold_payment, hand_mechanism, threshold_mechanism
from rn_regret import (
    regret_type_a_ascent, regret_type_a_hillclimb, regret_type_a_lambda_grid, regret_type_b, summarize,
)
from rn_types import C_HI, C_LO

RULE_LABELS = {
    "A": "A: Makespan + Pay-as-bid",
    "B": "B: Summe + VCG",
    "E": "E: Makespan + Clarke-Formel",
    "boost": "Boosted VCG (eigene Konstruktion)",
    "threshold": "Schwellenwert (Archer-Tardos, gestutzt)",
    "learned": "Gelernt (randomisiert)",
    "learned_rounded": "Gelernt, gerundet (argmax)",
}


def hand(rule, gamma=0.0):
    return lambda world, reports: hand_mechanism(world, reports, rule, gamma)


def basic_metrics(world, mech, reports):
    """Lücke, Zahlung/Kosten, IR-Verletzungen bei ehrlichem Bieten."""
    pi, pay = mech(world, reports)
    C = world.costs(reports)
    exp_cost = np.einsum("np,npk->nk", pi, C)
    exp_makespan = (pi * C.max(-1)).sum(1)
    opt = C.max(-1).min(1)
    utility = pay - exp_cost
    return {
        "gap_pct": float((exp_makespan.mean() / opt.mean() - 1.0) * 100.0),
        "pay_cost": float(pay.sum() / max(exp_cost.sum(), 1e-12)),
        "ir_share": float((utility < -1e-9).mean()),
        "argmax_gap_pct": float((np.take_along_axis(C.max(-1), pi.argmax(1)[:, None], 1).mean() / opt.mean() - 1.0) * 100.0),
    }


def evaluate(world, mech, reports, rng=None, mode="auto", learned=None):
    """Kennzahlen inkl. Regret. mode: "b_exact" (Typ B, 1-D-Gitter), "a_hillclimb" (Handregeln, Typ A), "a_ascent" (gelernt, Typ A)."""
    result = basic_metrics(world, mech, reports)
    rng = rng if rng is not None else np.random.default_rng(1)
    if world.kind == "B":
        regret, _ = regret_type_b(world, mech, reports, 200)
        result["regret_grid"] = None
    elif mode == "a_ascent":
        regret = regret_type_a_ascent(world, learned, reports, rng)
        result["regret_grid"] = float(regret_type_a_lambda_grid(world, learned, reports).mean())
    else:
        regret = regret_type_a_hillclimb(world, mech, reports, rng)
        result["regret_grid"] = float(regret_type_a_lambda_grid(world, mech, reports).mean())
    summary = summarize(regret)
    result.update(regret=summary["mean"], regret_max=summary["max"], regret_share=summary["share_above"])
    return result


def threshold_regret_sample(world, c, grid_points=40):
    """Numerischer Wahrhaftigkeits-Test der exakten Schwelle auf einer kleinen Stichprobe: größter Lügengewinn. Kostet Rechenzeit
    (jede Zahlung ist ein exaktes Integral), deshalb nur wenige Stichproben."""
    worst = 0.0
    grid = np.linspace(C_LO, C_HI, grid_points)
    S = world.S
    for row in c:
        for a in range(world.k):
            C_true = row[None, :] * S
            p_truth = _choose(S, row)
            u_truth = _threshold_payment(S, row, a, p_truth) - row[a] * S[p_truth, a]
            for g in grid:
                lie = row.copy()
                lie[a] = g
                p_lie = _choose(S, lie)
                u = _threshold_payment(S, lie, a, p_lie) - row[a] * S[p_lie, a]
                worst = max(worst, u - u_truth)
    return worst


def _choose(S, row):
    C = row[None, :] * S
    return int(_argmin_lex(C.max(-1)[None], C.sum(-1)[None])[0])


def comparison_table(world, reports, learned=None, gamma=3.0, with_threshold=True, rng=None):
    """Alle Mechanismen auf denselben Test-Meldungen -> Liste von Zeilen (dict mit 'key', 'label' und den Kennzahlen)."""
    rng = rng if rng is not None else np.random.default_rng(1)
    rows = []
    entries = [("A", hand("A")), ("B", hand("B")), ("E", hand("E")), ("boost", hand("boost", gamma))]
    if world.kind == "B" and with_threshold:
        entries.append(("threshold", lambda w, r: threshold_mechanism(w, r)))
    for key, mech in entries:
        if key == "threshold":
            metrics = basic_metrics(world, mech, reports[:60])
            metrics.update(regret=0.0, regret_max=0.0, regret_share=0.0, regret_grid=None, exact_truthful=True)
        else:
            metrics = evaluate(world, mech, reports, rng)
        rows.append({"key": key, "label": RULE_LABELS[key] + (f", γ = {gamma:g}" if key == "boost" else ""), **metrics})
    if learned is not None:
        rows.append({"key": "learned", "label": RULE_LABELS["learned"],
                     **evaluate(world, learned, reports, rng, "a_ascent", learned)})
        rounded = learned.rounded_copy()             # nicht differenzierbar: Typ A per Zufalls-Hill-Climbing statt Gradient
        rows.append({"key": "learned_rounded", "label": RULE_LABELS["learned_rounded"],
                     **evaluate(world, rounded, reports, rng, "a_hillclimb", None)})
    return rows


TRUTHFUL_KEYS = ("threshold", "B", "boost")       # exakt wahrheitsgetreue Mechanismen (Regret 0 per Konstruktion)


def verdict(rows):
    """Verdict-Kaskade (Warnungen zuerst) -> (Stufe, Code, Daten)."""
    by = {r["key"]: r for r in rows}
    learned = by.get("learned")
    if learned is None:
        return "info", "no_learned", {}
    data = {"gap": learned["gap_pct"], "regret": learned["regret"], "pay_cost": learned["pay_cost"]}
    if learned["regret"] > 1.0:
        return "warning", "regret_high", data
    for key in TRUTHFUL_KEYS:                          # ein wahrheitsgetreuer Mechanismus, der auf beiden Achsen mindestens so gut ist
        row = by.get(key)
        if row is not None and row["gap_pct"] <= learned["gap_pct"] + 0.5 and row["pay_cost"] <= learned["pay_cost"] + 0.05:
            return "warning", "dominated_by_truthful", data | {"label": row["label"], "t_gap": row["gap_pct"], "t_pay": row["pay_cost"]}
    return "success", "learned_ok", data


def rounding_note(rows):
    """Wenn Runden auf die wahrscheinlichste Zuteilung den Regret deutlich erhöht: (Regret randomisiert, Regret gerundet), sonst None."""
    by = {r["key"]: r for r in rows}
    learned, rounded = by.get("learned"), by.get("learned_rounded")
    if learned is None or rounded is None:
        return None
    if rounded["regret"] > 3 * max(learned["regret"], 0.02) and rounded["regret"] > learned["regret"] + 0.15:
        return learned["regret"], rounded["regret"]
    return None
