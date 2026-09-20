"""Kleine numpy-Bausteine: ein tanh-MLP mit manuellem Backprop (auch bis zur Eingabe - für die Fehlmeldungssuche in Typ A) und Adam.
Kein torch - das Portfolio bleibt numpy-only und damit auf Streamlit Cloud lauffähig. Stil wie `mappo-demo/mappo_nets.py`."""

import numpy as np


def softmax(logits):
    z = logits - logits.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class MLP:
    """tanh-MLP, Eingabe [N, d] -> Ausgabe [N, out]. Letzte Schicht klein initialisiert (Start ≈ gleichverteilte Zuteilung)."""

    def __init__(self, sizes, rng):
        self.W, self.b = [], []
        last = len(sizes) - 1
        for i in range(last):
            w = rng.standard_normal((sizes[i], sizes[i + 1])) / np.sqrt(sizes[i])
            if i == last - 1:
                w = w * 0.1
            self.W.append(w)
            self.b.append(np.zeros(sizes[i + 1]))

    def params(self):
        return self.W + self.b

    def forward(self, x):
        hs = [x]
        last = len(self.W) - 1
        for i in range(len(self.W)):
            z = hs[-1] @ self.W[i] + self.b[i]
            hs.append(np.tanh(z) if i < last else z)
        return hs

    def backward(self, hs, d_out, want_input_grad=False):
        """Gradienten (Gewichte + Biases in der Reihenfolge von `params()`); optional dL/dEingabe."""
        layers = len(self.W)
        gw, gb = [None] * layers, [None] * layers
        d = d_out
        for i in range(layers - 1, -1, -1):
            gw[i] = hs[i].T @ d
            gb[i] = d.sum(0)
            if i > 0 or want_input_grad:
                d = (d @ self.W[i].T) * ((1 - hs[i] ** 2) if i > 0 else 1.0)
        grads = gw + gb
        return (grads, d) if want_input_grad else grads


class Adam:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps

    def step(self, grads, max_norm=5.0):
        if max_norm:
            norm = np.sqrt(sum((g * g).sum() for g in grads))
            if norm > max_norm:
                grads = [g * (max_norm / (norm + 1e-12)) for g in grads]
        self.t += 1
        for i, g in enumerate(grads):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            self.params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
