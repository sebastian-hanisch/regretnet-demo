"""Regret-Schätzer. Regret_a = E_Instanz[max(0, max_Fehlmeldung u_a(Fehlmeldung) − u_a(wahr))] bei ehrlichen anderen, in Minuten.

WICHTIG: jeder Schätzer ist nur eine UNTERGRENZE des echten Regrets - er ist so stark wie sein Gegenspieler. Typ B hat einen
Parameter, dort ist die 1-D-Best-Response auf einem feinen Gitter praktisch exakt. In Typ A (Vektor-Typ) liefern die einfachen
Verfahren sehr verschiedene Werte: das gleichmäßige λ-Gitter aus auction-demo (alle eigenen Gebote mit demselben Faktor)
findet beim gelernten Netz fast nichts - genau das verteidigt das Training -, die Gradientensuche über Bündel-Multiplikatoren
deutlich mehr."""

import numpy as np

from rn_nets import sigmoid, softmax, softplus
from rn_types import C_HI, C_LO, SC

LAMBDA_LO, LAMBDA_HI = 0.5, 3.0
LAMBDA_GRID = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0, 3.0)


def _agent_utility(world, mech, reports, true_costs, agent):
    """Nutzen von `agent` in jeder Zeile: Zahlung − erwartete wahre Kosten (Zeilen = Meldungsprofile). -> [N]"""
    pi, pay = mech(world, reports)
    return pay[:, agent] - (pi * true_costs[:, :, agent]).sum(1)


def regret_type_b(world, mech, c, grid_points=200):
    """Exakte 1-D-Best-Response (feines Gitter über den Typbereich). -> (regret [N, k], bestes c' [N, k])"""
    true_costs = world.costs(c)
    N, k = c.shape
    grid = np.linspace(C_LO, C_HI, grid_points)
    regret = np.zeros((N, k))
    best = c.copy()
    for a in range(k):
        u_true = _agent_utility(world, mech, c, true_costs, a)
        u_best, arg = u_true.copy(), c[:, a].copy()
        for g in grid:
            u = _agent_utility(world, mech, world.replace(c, a, np.full(N, g)), true_costs, a)
            better = u > u_best + 1e-12
            u_best = np.where(better, u, u_best)
            arg = np.where(better, g, arg)
        regret[:, a] = np.maximum(u_best - u_true, 0.0)
        best[:, a] = arg
    return regret, best


def regret_type_a_lambda_grid(world, mech, bids, lambdas=LAMBDA_GRID):
    """Untergrenze aus auction-demo: alle eigenen Gebote mit demselben Faktor λ. -> regret [N, k]"""
    true_costs = world.costs(bids)
    N, k = len(bids), world.k
    regret = np.zeros((N, k))
    for a in range(k):
        u_true = _agent_utility(world, mech, bids, true_costs, a)
        u_best = u_true.copy()
        for lam in lambdas:
            rep = bids.copy()
            rep[:, a, 1:] = bids[:, a, 1:] * lam
            u_best = np.maximum(u_best, _agent_utility(world, mech, rep, true_costs, a))
        regret[:, a] = np.maximum(u_best - u_true, 0.0)
    return regret


def regret_type_a_hillclimb(world, mech, bids, rng, restarts=4, iters=40):
    """Ableitungsfreie Suche über Bündel-Multiplikatoren für BELIEBIGE Mechanismen (Handregeln): startet beim besten uniformen λ,
    dazu Zufallsstarts, dann multiplikatives Zufalls-Hill-Climbing. -> regret [N, k]"""
    true_costs = world.costs(bids)
    N, k = len(bids), world.k
    T1 = world.total - 1
    regret = np.zeros((N, k))
    for a in range(k):
        base = bids[:, a, 1:]

        def utility(m):
            rep = bids.copy()
            rep[:, a, 1:] = base * m
            return _agent_utility(world, mech, rep, true_costs, a)

        u_true = _agent_utility(world, mech, bids, true_costs, a)
        best_u, best_m = u_true.copy(), np.ones((N, T1))
        starts = [np.full((N, T1), lam) for lam in LAMBDA_GRID] + [rng.uniform(0.6, 2.0, (N, T1)) for _ in range(restarts)]
        for m0 in starts:
            u0 = utility(m0)
            better = u0 > best_u
            best_u, best_m = np.where(better, u0, best_u), np.where(better[:, None], m0, best_m)
        cur_u, cur_m = best_u.copy(), best_m.copy()
        for it in range(iters):
            sigma = 0.4 * (0.92 ** it) + 0.02
            cand = np.clip(cur_m * np.exp(sigma * rng.standard_normal((N, T1))), LAMBDA_LO, LAMBDA_HI)
            u = utility(cand)
            better = u > cur_u
            cur_u, cur_m = np.where(better, u, cur_u), np.where(better[:, None], cand, cur_m)
        regret[:, a] = np.maximum(np.maximum(best_u, cur_u) - u_true, 0.0)
    return regret


def ascent_type_a(world, lm, bids, agent, rng, steps, restarts, lo=LAMBDA_LO, hi=LAMBDA_HI):
    """Gradientenaufstieg über die Bündel-Multiplikatoren von `agent` für das gelernte Netz (RegretNet-Schätzer): normierter
    Gradientenschritt, mehrere Restarts. -> (bester Nutzen [N], bestes m [N, 2^n − 1], Nutzen bei Wahrheit [N])"""
    N = len(bids)
    T1 = world.total - 1
    masks_a = world.masks[:, agent]
    onehot = np.zeros((world.P, world.total))
    onehot[np.arange(world.P), masks_a] = 1.0
    C_true_a = bids[:, agent, :][:, masks_a]                      # [N, P]
    base = bids[:, agent, 1:]
    block = slice(agent * T1, (agent + 1) * T1)

    def run(m):
        rep = bids.copy()
        rep[:, agent, 1:] = base * m
        x = world.encode(rep)
        hs = lm.net.forward(x)
        out = hs[-1]
        pi = softmax(out[:, :world.P])
        r = out[:, world.P + agent]
        g = rep[:, agent, :][:, masks_a] - C_true_a               # [N, P]
        u = (pi * g).sum(1) + softplus(r)
        return u, pi, g, r, hs

    u_true, *_ = run(np.ones((N, T1)))
    best_u, best_m = u_true.copy(), np.ones((N, T1))
    for restart in range(restarts):
        m = np.ones((N, T1)) if restart == 0 else rng.uniform(0.6, 2.0, (N, T1))
        for it in range(steps + 1):
            u, pi, g, r, hs = run(m)
            better = u > best_u
            best_u, best_m = np.where(better, u, best_u), np.where(better[:, None], m, best_m)
            if it == steps:
                break
            d_out = np.zeros_like(hs[-1])
            d_out[:, :world.P] = pi * (g - (pi * g).sum(1, keepdims=True))
            d_out[:, world.P + agent] = sigmoid(r)
            _, dx = lm.net.backward(hs, d_out, want_input_grad=True)
            direct = (pi @ onehot)[:, 1:]                         # ∂u/∂(gemeldetes Gebot) direkt über die Zahlung
            dm = (dx[:, block] / SC + direct) * base
            eta = 0.35 * (0.93 ** it) + 0.02
            m = np.clip(m + eta * dm / (np.abs(dm).max(1, keepdims=True) + 1e-9), lo, hi)
    return best_u, best_m, u_true


def regret_type_a_ascent(world, lm, bids, rng, steps=60, restarts=6):
    """Gradientensuche für ein gelerntes Netz, danach auch das uniforme λ-Gitter (Maximum beider). -> regret [N, k]"""
    regret = np.zeros((len(bids), world.k))
    for a in range(world.k):
        best_u, _, u_true = ascent_type_a(world, lm, bids, a, rng, steps, restarts)
        regret[:, a] = np.maximum(best_u - u_true, 0.0)
    return np.maximum(regret, regret_type_a_lambda_grid(world, lm, bids))


def summarize(regret, threshold=0.1):
    return {"mean": float(regret.mean()), "max": float(regret.max()), "share_above": float((regret > threshold).mean())}


__all__ = ["regret_type_b", "regret_type_a_lambda_grid", "regret_type_a_hillclimb", "ascent_type_a", "regret_type_a_ascent", "summarize"]
