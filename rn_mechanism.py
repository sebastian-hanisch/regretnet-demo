"""Der gelernte Mechanismus: ein tanh-MLP bildet das Meldungsprofil auf (Zuteilung, Zahlung) ab.

Zuteilung: Softmax über alle k^n Zuteilungen (randomisiert!). Zahlung: pay_a = Σ_p π_p · (gemeldete Kosten von Agent a in Zuteilung p) +
softplus(r_a) - die erwarteten gemeldeten Kosten plus eine strikt positive Rente r_a (ex-interim individuell rational bezüglich der
Meldungen: bei ehrlichem Bieten ist der Nutzen = die Rente ≥ 0). Das schränkt nichts ein: jeder IR-Mechanismus zahlt mindestens die
erwarteten Kosten, seine Rente ist eine freie, nichtnegative Funktion der Meldungen. Ex-post-IR bei einer ausgewürfelten Zuteilung
gilt NICHT (Limitation).

Nutzen von Agent a mit wahren Kosten C_true bei Meldung R: u_a = Σ_p π_p (C_rep[p, a] − C_true[p, a]) + softplus(r_a)."""

import json

import numpy as np

from rn_nets import MLP, sigmoid, softmax, softplus
from rn_types import WorldA, WorldB

RENT_BIAS = -3.0


def make_net(world, hidden, rng):
    net = MLP([world.dim, hidden, hidden, world.P + world.k], rng)
    net.b[-1][world.P:] = RENT_BIAS
    return net


class LearnedMechanism:
    def __init__(self, world, net, rounded=False):
        self.world, self.net, self.rounded = world, net, rounded

    def outputs(self, reports):
        hs = self.net.forward(self.world.encode(reports))
        out = hs[-1]
        return out[:, :self.world.P], out[:, self.world.P:], hs

    def __call__(self, world, reports):
        logits, r, _ = self.outputs(reports)
        pi = softmax(logits)
        C = world.costs(reports)
        if self.rounded:
            index = pi.argmax(1)
            pi = np.zeros_like(pi)
            pi[np.arange(len(pi)), index] = 1.0
        pay = np.einsum("np,npk->nk", pi, C) + softplus(r)
        return pi, pay

    def rounded_copy(self):
        return LearnedMechanism(self.world, self.net, rounded=True)


# --- Speichern / Laden (kleine npz-Dateien) -----------------------------------------------------------

def save_mechanism(path, world, net, meta):
    arrays = {f"W{i}": w for i, w in enumerate(net.W)}
    arrays.update({f"b{i}": b for i, b in enumerate(net.b)})
    meta = dict(meta, kind=world.kind, n=world.n, k=world.k, layers=len(net.W))
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)


def load_mechanism(path):
    """-> (LearnedMechanism, World, meta)"""
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    world = WorldB(meta["n"], meta["k"]) if meta["kind"] == "B" else WorldA(meta["n"], meta["k"])
    net = MLP([world.dim, 1, 1, 1], np.random.default_rng(0))       # Form wird sofort überschrieben
    net.W = [data[f"W{i}"] for i in range(meta["layers"])]
    net.b = [data[f"b{i}"] for i in range(meta["layers"])]
    return LearnedMechanism(world, net), world, meta


__all__ = ["LearnedMechanism", "make_net", "save_mechanism", "load_mechanism", "sigmoid"]
