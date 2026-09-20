"""Training nach RegretNet: Augmented Lagrangian. Verlust = E[Makespan] + μ·E[Σ Rente] + Σ_a (λ_a·rgt_a + ρ/2·rgt_a²),
wobei rgt_a der (geschätzte) Regret von Agent a ist; λ_a += ρ·rgt_a alle T Schritte, ρ wächst bis ρmax. Der Regret wird an der
gefundenen Fehlmeldung ausgewertet, die Fehlmeldung selbst wird beim Ableiten als konstant behandelt (Envelope-Gradient).
Die ersten `warmup` Schritte laufen ohne Regret-Term.

Typ B: die Fehlmeldung ist ein einziger Kostenfaktor, gesucht auf einem Gitter. Typ A: ein Vektor von Bündel-Multiplikatoren,
gesucht per normiertem Gradientenaufstieg (`rn_regret.ascent_type_a`)."""

import time
from dataclasses import dataclass, replace

import numpy as np

from rn_mechanism import LearnedMechanism, make_net
from rn_nets import Adam, sigmoid, softmax, softplus
from rn_regret import ascent_type_a
from rn_types import C_HI, C_LO


@dataclass(frozen=True)
class TrainConfig:
    hidden: int = 32
    batch: int = 128
    steps: int = 2000
    grid: int = 12              # Typ B: Gitterpunkte der Fehlmeldungssuche im Training
    ascent_steps: int = 8       # Typ A: Aufstiegsschritte je Restart
    restarts: int = 2           # Typ A: Restarts
    mu: float = 0.1             # Gewicht der Rente im Designer-Ziel
    rho0: float = 1.0
    rho_max: float = 100.0
    rho_growth: float = 1.6
    lam_max: float = 1e9        # Obergrenze der Multiplikatoren (Schutz gegen frühes Aufschaukeln)
    kappa_max: float = 10.0     # Obergrenze des Regret-Preises κ_a = λ_a + ρ·rgt_a (das Pareto-Knopf-Äquivalent zu ρmax)
    ramp: float = 0.4           # Anteil der Schritte, über die κ von 0 auf seinen Wert hochgefahren wird
    update_every: int = 40
    warmup: int = 150
    lr0: float = 2e-2
    lr1: float = 1e-3
    seed: int = 0


QUICK_B = TrainConfig(hidden=48, batch=128, steps=2500, kappa_max=3.0)              # ~10 s lokal (Typ B, k=2, n=4)
DEFAULT_B = replace(QUICK_B, hidden=64, batch=256, steps=5000)                      # Vorberechnung
DEFAULT_A = TrainConfig(hidden=64, batch=256, steps=4000, ascent_steps=8, restarts=1, kappa_max=3.0)


@dataclass
class TrainResult:
    mechanism: LearnedMechanism
    history: dict
    seconds: float
    config: TrainConfig


def _schedule(cfg, step):
    frac = step / max(1, cfg.steps - 1)
    return cfg.lr0 * (cfg.lr1 / cfg.lr0) ** frac


def _lagrange(cfg, state, rgt_sum, count):
    """Alle T Schritte: Multiplikatoren nachziehen und ρ erhöhen."""
    mean = rgt_sum / max(1, count)
    state["lam"] = np.minimum(state["lam"] + state["rho"] * mean, cfg.lam_max)
    state["rho"] = min(state["rho"] * cfg.rho_growth, cfg.rho_max)


def train(world, cfg, progress=None):
    """Trainiert einen Mechanismus für die gegebene Welt (Typ B oder A)."""
    started = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    k, P = world.k, world.P
    net = make_net(world, cfg.hidden, rng)
    opt = Adam(net.params(), cfg.lr0)
    state = {"lam": np.zeros(k), "rho": cfg.rho0}
    rgt_sum, count = np.zeros(k), 0
    history = {"step": [], "makespan_gap": [], "rent": [], "regret": []}
    grid = np.linspace(C_LO, C_HI, cfg.grid)
    lm = LearnedMechanism(world, net)
    win_gap, win_rent, win_rgt = [], [], []

    for step in range(cfg.steps):
        opt.lr = _schedule(cfg, step)
        reports = world.sample(rng, cfg.batch)
        N = len(reports)
        x = world.encode(reports)
        hs = net.forward(x)
        out = hs[-1]
        pi = softmax(out[:, :P])
        r = out[:, P:]
        C_true = world.costs(reports)                                   # [N, P, k]
        mk = C_true.max(-1)                                             # [N, P]
        e_mk = (pi * mk).sum(1)

        rgt = np.zeros(k)
        found, kappa = [], np.zeros(k)
        if step >= cfg.warmup:
            u_true = softplus(r)                                                # ehrlich: Nutzen = Rente
            for a in range(k):                                                  # Phase 1: Fehlmeldungen suchen, Regret schätzen
                if world.kind == "B":
                    best_u, best_val = _search_b(world, net, reports, a, grid, u_true[:, a])
                    rep = world.replace(reports, a, best_val)
                else:
                    best_u, best_m, _ = ascent_type_a(world, lm, reports, a, rng, cfg.ascent_steps, cfg.restarts)
                    rep = reports.copy()
                    rep[:, a, 1:] = reports[:, a, 1:] * best_m
                imp = best_u > u_true[:, a] + 1e-12
                rgt[a] = float(np.mean(np.where(imp, best_u - u_true[:, a], 0.0)))
                found.append((rep, imp))
            ramp = min(1.0, (step - cfg.warmup) / max(1.0, cfg.ramp * (cfg.steps - cfg.warmup)))
            kappa = np.clip(state["lam"] + state["rho"] * rgt, 0.0, cfg.kappa_max) * ramp   # dL/drgt_a des Augmented Lagrangian
            rgt_sum += rgt
            count += 1
        _, grads = surrogate(world, net, reports, found, kappa, cfg.mu)         # Phase 2: Gradient an der Fehlmeldung
        opt.step(grads)

        if step >= cfg.warmup and (step - cfg.warmup + 1) % cfg.update_every == 0:
            _lagrange(cfg, state, rgt_sum, count)
            rgt_sum, count = np.zeros(k), 0
        win_gap.append(float(e_mk.mean() / mk.min(1).mean() - 1.0) * 100.0)
        win_rent.append(float(softplus(r).sum(1).mean()))
        win_rgt.append(float(rgt.mean()))
        if (step + 1) % 50 == 0 or step == cfg.steps - 1:
            history["step"].append(step + 1)
            history["makespan_gap"].append(float(np.mean(win_gap)))
            history["rent"].append(float(np.mean(win_rent)))
            history["regret"].append(float(np.mean(win_rgt)))
            win_gap, win_rent, win_rgt = [], [], []
            if progress is not None:
                progress(step + 1, cfg.steps, {k_: v[-1] for k_, v in history.items()})
    return TrainResult(lm, history, time.perf_counter() - started, cfg)


def surrogate(world, net, reports, found, kappa, mu):
    """Surrogat-Verlust bei FESTEN Fehlmeldungen `found` = [(gemeldetes Profil, verbessert?)] je Agent und festen Gewichten κ_a
    (Envelope-Gradient) und sein Gradient nach den Netzparametern. Verlust = E[Makespan] + μ·E[Σ Rente]
    + Σ_a κ_a · E[verbessert · (u(Fehlmeldung) − u(wahr))]."""
    k, P = world.k, world.P
    N = len(reports)
    hs = net.forward(world.encode(reports))
    out = hs[-1]
    pi = softmax(out[:, :P])
    r = out[:, P:]
    C_true = world.costs(reports)
    mk = C_true.max(-1)
    e_mk = (pi * mk).sum(1)
    loss = float(e_mk.mean() + mu * softplus(r).sum(1).mean())
    d_out = np.zeros_like(out)
    d_out[:, :P] = pi * (mk - e_mk[:, None]) / N
    d_out[:, P:] = mu * sigmoid(r) / N
    grads_m = None
    for a, (rep, imp) in enumerate(found):
        w = kappa[a] * imp / N
        hs_m = net.forward(world.encode(rep))
        out_m = hs_m[-1]
        pi_m = softmax(out_m[:, :P])
        g = world.costs(rep)[:, :, a] - C_true[:, :, a]
        u_m = (pi_m * g).sum(1) + softplus(out_m[:, P + a])
        loss += float((w * (u_m - softplus(r[:, a]))).sum())
        d_m = np.zeros_like(out_m)
        d_m[:, :P] = w[:, None] * pi_m * (g - (pi_m * g).sum(1, keepdims=True))
        d_m[:, P + a] = w * sigmoid(out_m[:, P + a])
        gm = net.backward(hs_m, d_m)
        grads_m = gm if grads_m is None else [x1 + x2 for x1, x2 in zip(grads_m, gm)]
        d_out[:, P + a] -= w * sigmoid(r[:, a])
    g_true = net.backward(hs, d_out)
    grads = g_true if grads_m is None else [a1 + a2 for a1, a2 in zip(g_true, grads_m)]
    return loss, grads


def _search_b(world, net, c, agent, grid, u_true):
    """Beste Fehlmeldung von `agent` je Stichprobe auf dem Gitter (Typ B). -> (bester Nutzen [N], bester Wert c' [N])"""
    N = len(c)
    P = world.P
    best_u = u_true.copy()
    best_val = c[:, agent].copy()
    S_a = world.S[:, agent]
    for g in grid:
        rep = world.replace(c, agent, np.full(N, g))
        out = net.forward(world.encode(rep))[-1]
        pi = softmax(out[:, :P])
        u = (g - c[:, agent]) * (pi @ S_a) + softplus(out[:, P + agent])
        better = u > best_u
        best_u = np.where(better, u, best_u)
        best_val = np.where(better, g, best_val)
    return best_u, best_val
