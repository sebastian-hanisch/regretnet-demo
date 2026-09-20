"""Die "Welten" (Typmodelle) der RegretNet-Demo. Jede Welt beschreibt, was ein Agent privat weiß, wie aus den Meldungen Kosten je
Zuteilung entstehen und wie Meldungen für das Netz kodiert werden.

**Typ B (ein Parameter, Hauptstück):** feste Instanz (Aufträge und Startpositionen wie in der Vehikel-Demo), der Agent kennt einen
privaten Kostenfaktor c_a ~ U[C_LO, C_HI] und seine Kosten sind c_a · W_a(S) mit öffentlicher Bündelzeit W_a(S) (der Bündelzeit-
Tabelle der Auktions-Demo). Bei einem Parameter je Agent gibt es eine exakt wahrheitsgetreue Analytik (Schwellenwert-Mechanismus).

**Typ A (mehrere Parameter, sekundär):** der Typ ist der ganze Bündelkosten-Vektor eines Agenten (2^n − 1 Zahlen), abgeleitet aus einer
zufälligen Instanz (Positionen, Dauern). Die Agenten-Typen sind über die gemeinsame Instanz korreliert (Erwartung über die Instanz -
das entspricht dem Regret-Begriff bei ehrlichen anderen, nicht einem unabhängigen Bayes-Modell).

Alle Mechanismen arbeiten auf `costs(reports) -> [N, P, k]` (Kosten je Zuteilung und Agent bei gemeldeten Typen)."""

import numpy as np

import cn_constants as C
from auction_bids import bundle_bid_table
from cn_scenario import generate_instance
from rn_partitions import gather_costs, gather_costs_single, partitions

C_LO, C_HI = 0.5, 2.0                   # Typbereich des Kostenfaktors (Typ B)
SC = 30.0                                # Normierung der Netz-Eingabe (Minuten)
B_INSTANCE_SEED = C.DEFAULT_SEED         # feste Instanz von Typ B (Seed 9, wie die Vehikel-Demos)
HELDOUT_SEED_BASE = 100_000              # feste Test-Seeds, nie ein Demo- oder Trainings-Seed
A_VAR, A_TRAVEL = 0.3, 1.0               # Streuung und Fahrzeit der Typ-A-Instanzen


def _hand_instance_k(n_agents):
    from cn_scenario import Instance
    base = hand_instance()
    starts = tuple((c + 0.5) * C.POSITION_RANGE_MAX / n_agents for c in range(n_agents))
    return Instance(n_jobs=2, n_agents=n_agents, jobs=base.jobs, agent_start_positions=starts, travel_time_per_unit=0.0)


def hand_instance():
    """Handbeispiel (n = 2): zwei gleiche Aufträge der Dauer 10 an derselben Stelle, keine Fahrzeit - jeder Agent braucht für ein Bündel
    10 min je Auftrag. So sind Zuteilung und Schwellenwerte von Hand herleitbar."""
    from cn_scenario import Instance, Job
    return Instance(n_jobs=2, n_agents=2, jobs=(Job(0, 10.0, 10.0), Job(1, 10.0, 10.0)), agent_start_positions=(5.0, 15.0),
                    travel_time_per_unit=0.0)


class WorldB:
    kind = "B"

    def __init__(self, n_jobs, n_agents, instance_seed=B_INSTANCE_SEED, var=A_VAR, travel=A_TRAVEL, instance=None):
        self.n, self.k = n_jobs, n_agents
        if instance is None:
            instance = _hand_instance_k(n_agents) if n_jobs == 2 else generate_instance(n_jobs, n_agents, var, travel, instance_seed)
        self.instance = instance
        self.table = bundle_bid_table(self.instance)                 # [k, 2^n] Bündelzeiten W_a(S)
        self.masks, self.sizes = partitions(n_jobs, n_agents)
        self.P = self.masks.shape[0]
        self.S = gather_costs_single(self.table, self.masks)         # [P, k] Bündelzeit je Zuteilung und Agent
        self.dim = n_agents

    def costs(self, reports):
        """reports = c [N, k] -> Kosten [N, P, k]."""
        return reports[:, None, :] * self.S[None, :, :]

    def encode(self, reports):
        return (reports - 0.5 * (C_LO + C_HI)) / (0.5 * (C_HI - C_LO))

    def sample(self, rng, n_samples):
        return rng.uniform(C_LO, C_HI, size=(n_samples, self.k))

    def heldout(self, n_samples, seed=HELDOUT_SEED_BASE):
        return np.random.default_rng(seed).uniform(C_LO, C_HI, size=(n_samples, self.k))

    def replace(self, reports, agent, values):
        """Meldung von `agent` in allen Zeilen durch `values` [N] ersetzen."""
        out = reports.copy()
        out[:, agent] = values
        return out

    def opt_makespan(self, reports):
        return self.costs(reports).max(-1).min(-1)


def bundle_bids_batch(positions, durations, starts, travel):
    """Vektorisierte geschlossene Bündelgebote: positions/durations [N, n], starts [k] -> [N, k, 2^n] (Spalte 0 = 0)."""
    N, n = positions.shape
    k = len(starts)
    total = 1 << n
    lo = np.full((N, total), np.inf)
    hi = np.full((N, total), -np.inf)
    dsum = np.zeros((N, total))
    for j in range(n):
        step = 1 << j
        lo[:, step:2 * step] = np.minimum(lo[:, :step], positions[:, j:j + 1])
        hi[:, step:2 * step] = np.maximum(hi[:, :step], positions[:, j:j + 1])
        dsum[:, step:2 * step] = dsum[:, :step] + durations[:, j:j + 1]
    lo[:, 0] = hi[:, 0] = 0.0
    out = np.zeros((N, k, total))
    for a in range(k):
        out[:, a] = dsum + travel * ((hi - lo) + np.minimum(np.abs(starts[a] - lo), np.abs(starts[a] - hi)))
    out[:, :, 0] = 0.0
    return out


class WorldA:
    kind = "A"

    def __init__(self, n_jobs, n_agents, var=A_VAR, travel=A_TRAVEL):
        self.n, self.k = n_jobs, n_agents
        self.var, self.travel = var, travel
        self.masks, self.sizes = partitions(n_jobs, n_agents)
        self.P = self.masks.shape[0]
        self.starts = np.array([(c + 0.5) * C.POSITION_RANGE_MAX / n_agents for c in range(n_agents)])
        self.total = 1 << n_jobs
        self.dim = n_agents * (self.total - 1)

    def costs(self, reports):
        """reports = Gebote [N, k, 2^n] -> Kosten [N, P, k]."""
        return gather_costs(reports, self.masks)

    def encode(self, reports):
        return reports[:, :, 1:].reshape(len(reports), -1) / SC

    def sample(self, rng, n_samples):
        n = self.n
        positions = rng.uniform(0, C.POSITION_RANGE_MAX, size=(n_samples, n))
        base = rng.uniform(C.DURATION_BASE_RANGE[0], C.DURATION_BASE_RANGE[1], size=(n_samples, n))
        spike = rng.random((n_samples, n)) < self.var * C.SPIKE_PROBABILITY_SCALE
        durations = np.where(spike, base * C.SPIKE_MULTIPLIER, base)
        return bundle_bids_batch(positions, durations, self.starts, self.travel)

    def heldout(self, n_samples, seed=HELDOUT_SEED_BASE):
        """Echte `generate_instance`-Instanzen mit festen Seeds (Test-Instanzen)."""
        return np.stack([
            bundle_bid_table(generate_instance(self.n, self.k, self.var, self.travel, seed + i)) for i in range(n_samples)
        ])

    def replace(self, reports, agent, values):
        out = reports.copy()
        out[:, agent, :] = values
        return out

    def opt_makespan(self, reports):
        return self.costs(reports).max(-1).min(-1)
