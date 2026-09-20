"""Zahlungsregeln und strategisches Bieten (Beschaffungs-/Reverse-Auktion): der Auktionator kauft die Auftragsbearbeitung, jeder
Agent hat als KOSTEN seine eigene Fertigstellungszeit (Modellannahme: 1 Geldeinheit = 1 min; dass ein Agent lügen könnte,
ist Modellierung, keine Eigenschaft des Fahrzeugs - im Vehikel sind die Kosten öffentlich).

Vier handentworfene Regeln (Ziel der Zuteilung x Zahlung):
  A  Makespan-Zuteilung + Pay-as-bid          (das, was das Fahrzeug wirklich will, aber der Bieter kann aufblähen)
  B  Summen-Zuteilung   + VCG (Clarke)        (wahrheitsgetreu - aber minimiert die Gesamtzeit, nicht den Makespan)
  C  Summen-Zuteilung   + Pay-as-bid
  E  Makespan-Zuteilung + Clarke-Formel       (VCG-Formel auf die Makespan-Zuteilung "angeklebt": Zahlung kann unter Kosten fallen)
  D  Makespan-Zuteilung + Schwellenwert-Zahlung (nur im Sweep/README: zweitbester Makespan ohne das Gewinner-Bündel)

VCG ist nur für das SUMMENZIEL wahrheitsgetreu (Effizienz = Wohlfahrtsmaximierung). Für das Makespan-Ziel gibt es keine
solche Regel in diesem Katalog - genau diese Lücke motiviert die spätere Fortsetzung (gelerntes Mechanism-Design)."""

from dataclasses import dataclass

import numpy as np

from auction_bids import INF
from auction_wd import winner_determination

RULES = {
    "A": ("max", "pay_as_bid"),
    "B": ("sum", "vcg"),
    "C": ("sum", "pay_as_bid"),
    "D": ("max", "threshold"),
    "E": ("max", "clarke_formula"),
}
RULE_LABELS = {
    "A": "A: Makespan-Zuteilung + Pay-as-bid",
    "B": "B: Summen-Zuteilung + VCG",
    "C": "C: Summen-Zuteilung + Pay-as-bid",
    "D": "D: Makespan-Zuteilung + Schwellenwert",
    "E": "E: Makespan-Zuteilung + Clarke-Formel",
}


@dataclass(frozen=True)
class Outcome:
    rule: str
    value: float            # Zielwert der Zuteilung (Makespan bzw. Gesamtzeit) auf den GEMELDETEN Geboten
    masks: tuple            # Bündel je Agent
    payments: tuple         # Zahlung je Agent (0 für Verlierer; nan, wenn nicht definiert)
    undefined_agents: tuple  # Agenten, für die die Zahlung nicht definiert ist (Monopol: ohne sie keine zulässige Lösung)
    feasible: bool


def _without(bids, agent):
    """Der Agent nimmt nicht teil: nur das leere Bündel bleibt."""
    reduced = bids.copy()
    reduced[agent, 1:] = INF
    return reduced


def run_mechanism(bids, rule, payment_for=None):
    """Führt eine Regel auf den (gemeldeten) Geboten aus. `payment_for`: nur für diese Agenten Zahlungen berechnen
    (spart Rechenzeit bei den Entfernungs-DPs); Default: alle Gewinner."""
    objective, payment_kind = RULES[rule]
    k = bids.shape[0]
    wd = winner_determination(bids, objective)
    payments = [0.0] * k
    undefined = []
    if not wd.feasible:
        return Outcome(rule, INF, wd.masks, tuple(payments), (), False)
    winners = [a for a in range(k) if wd.masks[a] != 0]
    wanted = winners if payment_for is None else [a for a in winners if a in payment_for]
    total = sum(bids[a][wd.masks[a]] for a in winners)
    for a in wanted:
        own = float(bids[a][wd.masks[a]])
        if payment_kind == "pay_as_bid":
            payments[a] = own
        elif payment_kind in ("vcg", "clarke_formula"):
            without = winner_determination(_without(bids, a), "sum")
            if not without.feasible:
                payments[a] = float("nan")
                undefined.append(a)
            else:
                payments[a] = float(without.value - (total - own))
        elif payment_kind == "threshold":
            blocked = bids.copy()
            blocked[a, wd.masks[a]] = INF
            second = winner_determination(blocked, "max")
            payments[a] = float(second.value) if second.feasible else float("nan")
            if not second.feasible:
                undefined.append(a)
    return Outcome(rule, float(wd.value), wd.masks, tuple(payments), tuple(undefined), True)


def real_makespan(bids, masks):
    """Echter Makespan einer Zuteilung = Maximum der WAHREN Gebote der Zuschlagsbündel."""
    return float(max((bids[a][m] for a, m in enumerate(masks) if m != 0), default=0.0))


def true_cost(true_bids, outcome, agent):
    mask = outcome.masks[agent]
    return 0.0 if mask == 0 else float(true_bids[agent][mask])


def utility(true_bids, outcome, agent):
    """Nutzen = Zahlung - wahre Kosten des zugeteilten Bündels (Verlierer: 0). None, wenn die Zahlung nicht definiert ist."""
    if outcome.masks[agent] == 0:
        return 0.0
    payment = outcome.payments[agent]
    if payment != payment:            # nan
        return None
    return payment - true_cost(true_bids, outcome, agent)


def misreport(true_bids, agent, rule, factor):
    """Der Agent skaliert ALLE seine Gebote mit `factor` (Strategieraum: gleichmäßiges Aufblähen/Drücken). Gibt das Ergebnis
    auf den gemeldeten Geboten zurück; der Nutzen wird gegen die WAHREN Gebote bewertet."""
    reported = true_bids.copy()
    finite = np.isfinite(reported[agent])
    reported[agent, finite] = reported[agent, finite] * factor
    reported[agent, 0] = 0.0
    return run_mechanism(reported, rule, payment_for=(agent,))


def misreport_curve(true_bids, agent, rule, factors):
    """Für jeden Faktor: (Faktor, Nutzen, Zuteilung des Agenten (Bündelgröße), Zielwert der Zuteilung). Nutzen None, wenn
    nicht definiert."""
    rows = []
    for factor in factors:
        outcome = misreport(true_bids, agent, rule, factor)
        size = bin(outcome.masks[agent]).count("1") if outcome.feasible else 0
        rows.append((float(factor), utility(true_bids, outcome, agent) if outcome.feasible else None, size, outcome.value))
    return rows


def regret_of_agent(true_bids, agent, rule, factors):
    """Regret = größter Nutzengewinn einer Fehlmeldung gegenüber ehrlichem Bieten (Untergrenze: nur das Faktoren-Gitter)."""
    honest = misreport(true_bids, agent, rule, 1.0)
    honest_utility = utility(true_bids, honest, agent)
    if honest_utility is None:
        return None, None
    best_gain, best_factor = 0.0, 1.0
    for factor in factors:
        outcome = misreport(true_bids, agent, rule, factor)
        u = utility(true_bids, outcome, agent) if outcome.feasible else None
        if u is not None and u - honest_utility > best_gain + 1e-9:
            best_gain, best_factor = u - honest_utility, float(factor)
    return best_gain, best_factor


def payment_stats(true_bids, outcome):
    """Zahlung/Kosten, kleinste Rendite (Zahlung - Kosten; < 0 = Individuelle Rationalität verletzt), Monopol-Agenten."""
    winners = [a for a in range(len(outcome.masks)) if outcome.masks[a] != 0]
    if outcome.undefined_agents or not outcome.feasible:
        return {"defined": False, "ratio": None, "min_margin": None, "undefined": list(outcome.undefined_agents)}
    costs = [true_cost(true_bids, outcome, a) for a in winners]
    pays = [outcome.payments[a] for a in winners]
    total_cost = sum(costs)
    return {
        "defined": True, "ratio": (sum(pays) / total_cost) if total_cost > 0 else None,
        "min_margin": min(p - c for p, c in zip(pays, costs)) if winners else 0.0, "undefined": [],
    }
