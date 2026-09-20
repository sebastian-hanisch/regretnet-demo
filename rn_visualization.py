"""Plotly-Figuren der RegretNet-Demo. Alle Figuren laufen durch `lock_axes` (Touch-Scrolling-Konvention des Portfolios)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cn_visualization import lock_axes
from rn_types import C_HI, C_LO

COLORS = {
    "A": "#D55E00", "B": "#0072B2", "E": "#E69F00", "boost": "#009E73", "threshold": "#000000",
    "learned": "#CC79A7", "learned_rounded": "#999999", "family": "#56B4E9",
}


def build_pareto(rows, family=None):
    """Lücke zum Optimum (x) gegen Überzahlung (y); Punktgröße = Regret. Ohne Regret (exakt wahrheitsgetreu) ist der Punkt klein."""
    fig = go.Figure()
    for row in rows:
        size = 10 + 9 * np.sqrt(max(row["regret"], 0.0))
        fig.add_trace(go.Scatter(
            x=[np.log1p(max(row["gap_pct"], 0.0))], y=[row["pay_cost"]], mode="markers+text", text=[row["label"].split(":")[0].split(" (")[0]],
            textposition="top center", showlegend=False,
            marker=dict(size=size, color=COLORS.get(row["key"], "#888"), line=dict(color="black", width=1), opacity=0.85),
            hovertext=f"{row['label']}: Lücke {row['gap_pct']:.1f} %, Zahlung/Kosten {row['pay_cost']:.2f}, Regret {row['regret']:.2f} min",
            hoverinfo="text",
        ))
    if family:
        fig.add_trace(go.Scatter(
            x=[np.log1p(max(f["gap_pct"], 0.0)) for f in family], y=[f["pay_cost"] for f in family], mode="markers",
            name="gelernt (vorberechnet)",
            marker=dict(size=[10 + 9 * np.sqrt(f["regret"]) for f in family], color=COLORS["family"], opacity=0.6,
                        line=dict(color="black", width=1)),
            hovertext=[f"{f['label']}: Lücke {f['gap_pct']:.1f} %, Zahlung/Kosten {f['pay_cost']:.2f}, Regret {f['regret']:.2f} min"
                       for f in family], hoverinfo="text",
        ))
    ticks = [0, 1, 2, 5, 10, 20, 50, 100]
    fig.update_xaxes(title="erwarteter Makespan über dem Optimum (%, log(1+x)-Skala)", tickvals=[np.log1p(t) for t in ticks],
                     ticktext=[str(t) for t in ticks], range=[-0.15, np.log1p(max([r["gap_pct"] for r in rows] + [5.0])) + 0.4])
    fig.update_yaxes(title="Zahlung / Kosten")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_training_curves(history):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Makespan über Optimum (%)", "Rente (min)", "Regret im Training (min)"))
    for col, key in enumerate(("makespan_gap", "rent", "regret"), start=1):
        fig.add_trace(go.Scatter(x=history["step"], y=history[key], mode="lines", line=dict(color=COLORS["learned"], width=3),
                                 showlegend=False), row=1, col=col)
    fig.update_xaxes(title_text="Schritt")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_misreport_b(grid, curves, true_value, agent):
    """Nutzen von `agent` gegen die gemeldete Kostenrate c' (eine Linie je Mechanismus); ehrlich = gestrichelt."""
    fig = go.Figure()
    for key, label, values in curves:
        fig.add_trace(go.Scatter(x=grid, y=values, mode="lines", name=label, line=dict(color=COLORS.get(key, "#888"), width=3)))
    fig.add_vline(x=true_value, line_dash="dash", line_color="gray", annotation_text="ehrlich", annotation_position="top")
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_xaxes(title=f"gemeldeter Kostenfaktor c' von Agent {agent + 1}")
    fig.update_yaxes(title="Nutzen = Zahlung − wahre Kosten (min)")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_allocation_map(grid, learned_work, exact_work):
    """Erwartete Arbeit (Bündelzeit) von Agent 1 über (c1, c2): gelernt (glatt) gegen exakt (Stufenfunktion)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("gelernt (randomisiert)", "exakte Schwelle (deterministisch)"))
    zmax = float(max(learned_work.max(), exact_work.max()))
    for col, z in enumerate((learned_work, exact_work), start=1):
        fig.add_trace(go.Heatmap(x=grid, y=grid, z=z, zmin=0, zmax=zmax, colorscale="Viridis", showscale=col == 2,
                                 colorbar=dict(title="Arbeit A1 (min)") if col == 2 else None), row=1, col=col)
    fig.update_xaxes(title_text="c₁ (Agent 1)")
    fig.update_yaxes(title_text="c₂ (Agent 2)", col=1)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_misreport_a(lambdas, curves, agent):
    """Typ A: Nutzen gegen den Faktor λ, mit dem der Agent ALLE eigenen Gebote skaliert (uniformes Gitter)."""
    fig = go.Figure()
    for key, label, values in curves:
        fig.add_trace(go.Scatter(x=lambdas, y=values, mode="lines+markers", name=label, line=dict(color=COLORS.get(key, "#888"), width=3)))
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="ehrlich", annotation_position="top")
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_xaxes(title=f"Faktor λ, mit dem Agent {agent + 1} alle seine Gebote skaliert")
    fig.update_yaxes(title="mittlerer Nutzen (min)")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_wall_chart(rows):
    """Die Wand bei n = 4: Lücke des Allokations-Netzes OHNE Anreize (nur Makespan minimieren) je (n, k)."""
    labels = [f"n={r['n']}, k={r['k']}" for r in rows]
    fig = go.Figure(go.Bar(x=labels, y=[r["plain_gap"] for r in rows], marker_color=COLORS["learned"],
                           text=[f"{r['plain_gap']:.1f} %" for r in rows], textposition="outside"))
    fig.update_yaxes(title="Lücke ohne jede Anreiz-Nebenbedingung (%)", rangemode="tozero")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
    return lock_axes(fig)


__all__ = ["build_pareto", "build_training_curves", "build_misreport_b", "build_allocation_map", "build_misreport_a",
           "build_wall_chart", "C_LO", "C_HI"]
