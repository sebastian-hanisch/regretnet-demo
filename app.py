"""
RegretNet: ein gelernter Mechanismus für die Kran-Auftragsvergabe – interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Achtes und letztes Stück der "Konzepte"-Reihe, Multi-Agenten-Koordinations-Linie - die Fortsetzung von auction-demo. Dort zeigte sich:
keine handentworfene Zahlungsregel erfüllt gleichzeitig Makespan-Optimum, Kostendeckung und Wahrhaftigkeit. Hier wird ein Mechanismus
gelernt (RegretNet, Dütting et al., ICML 2019) und ehrlich gegen die exakte Analytik gemessen. Ergebnis: teils negativ - gelernt
approximiert bekannte Analytik bei kleinen Größen, wird dort geschlagen und bricht bei n = 4 ein.
"""

import json
import time
from pathlib import Path

import numpy as np
import streamlit as st

import cn_constants as C
from cn_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from rn_baselines import hand_mechanism, threshold_mechanism, work_steps
from rn_evaluation import comparison_table, hand, rounding_note, verdict
from rn_mechanism import load_mechanism
from rn_regret import LAMBDA_GRID
from rn_train import QUICK_B, TrainConfig, train
from rn_types import C_HI, C_LO, WorldA, WorldB
from rn_visualization import (
    build_allocation_map,
    build_misreport_a,
    build_misreport_b,
    build_pareto,
    build_training_curves,
    build_wall_chart,
)

st.set_page_config(page_title="RegretNet – Sebastian Hanisch", layout="wide")

WEIGHTS = Path(__file__).resolve().parent / "weights"
KIND_LABELS = {"B": "B – ein Kostenfaktor je Agent (Hauptstück)", "A": "A – Bündelkosten-Vektor (sekundär)"}
LIVE_LABEL = "Live trainieren"


# --- Zwischenspeicher -------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _world(kind, n, k):
    return WorldB(n, k) if kind == "B" else WorldA(n, k)


@st.cache_resource(show_spinner=False)
def _precomputed(name):
    return load_mechanism(WEIGHTS / f"{name}.npz")


@st.cache_data(show_spinner=False)
def _manifest():
    path = WEIGHTS / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False)
def _wall():
    path = WEIGHTS / "wall.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_resource(show_spinner=False)
def _reports(kind, n, k):
    world = _world(kind, n, k)
    return world.heldout(C.N_TEST_B if kind == "B" else C.N_TEST_A)


@st.cache_data(show_spinner=False)
def _comparison(kind, n, k, mech_key, gamma, _mech):
    return comparison_table(_world(kind, n, k), _reports(kind, n, k), _mech, gamma=gamma)


def _fmt_int(n):
    return f"{n:,}".replace(",", ".")


def _fmt_seconds(seconds):
    return f"{seconds:.1f} s" if seconds >= 1 else f"{seconds * 1000:.0f} ms"


def _supported(kind, n, k):
    """Welche (Typ, n, k) gibt es? Live-Training nur bis k^n Zuteilungen = LIVE_MAX_PARTITIONS und nur Typ B."""
    live = kind == "B" and k ** n <= C.LIVE_MAX_PARTITIONS
    pre = [name for name, spec in C.WEIGHT_SPECS.items() if spec[:3] == (kind, n, k)]
    return live, pre


# --- Kopf --------------------------------------------------------------------------------------------

st.title("🧠 RegretNet: ein gelernter Mechanismus für die Kran-Auftragsvergabe")
st.info(
    "**Fortsetzung von auction-demo** - und ausdrücklich ein teils **negatives Ergebnis**. Dort zeigte sich: keine handentworfene "
    "Zahlungsregel erfüllt gleichzeitig Makespan-Optimum, Kostendeckung und Wahrhaftigkeit. Hier lernt ein Netz Zuteilung und Zahlung "
    "(RegretNet, Dütting et al., ICML 2019) - und wird gegen die exakte Analytik gemessen. Gemessen: bei **einem** privaten Parameter "
    "je Agent gibt es eine exakt wahrheitsgetreue Analytik, die das gelernte Netz auf beiden Achsen schlägt; bei **Bündelkosten-Vektoren** "
    "approximiert das Netz nur wahrheitsgetreue Varianten von VCG und bricht ab n = 4 ein."
)
st.markdown(
    """
Ein Agent kennt seine Kosten privat und kann bei der Meldung lügen. Der **Regret** misst, wie viel er dadurch im besten Fall gewinnt
(Minuten, bei ehrlichen anderen). Ein Mechanismus mit Regret 0 ist wahrheitsgetreu. Das Netz bekommt das Meldungsprofil und gibt eine
**Zuteilung** (Wahrscheinlichkeiten über alle Zuteilungen der Aufträge) und **Zahlungen** aus; trainiert wird der erwartete Makespan
unter der Nebenbedingung "Regret klein". Als Gegenprobe stehen die Handregeln aus auction-demo und - im Ein-Parameter-Fall - die exakte
Schwellenwert-Analytik daneben.
"""
)
st.caption(
    "Kernaussage: das gelernte Netz kommt bei kleinen Größen nah an bekannte Analytik heran, wird aber von ihr geschlagen (Lücke, "
    "Überzahlung und Regret), der Regret ist nur als Untergrenze messbar, und ohne Rundung bleibt die Zuteilung randomisiert."
)

with st.expander("Wie funktioniert diese Demo?", expanded=True):
    st.markdown(
        r"""
**Modell (Beschaffung, wie in auction-demo).** Der Auktionator kauft die Bearbeitung; die Kosten eines Agenten sind seine Fertigstellungszeit,
1 Geldeinheit = 1 Minute. Dass er lügen könnte, ist Modellierung. Ziel ist der **Makespan** (Maximum der Kosten der Agenten).

**Typ B - ein Parameter.** Die Aufträge sind fest; jeder Agent kennt einen privaten Kostenfaktor $c_a \in [0.5, 2]$, seine Kosten sind
$c_a \cdot W_a(S)$ mit öffentlicher Bündelzeit $W_a(S)$. Dann gilt (Archer & Tardos, FOCS 2001, "Truthful mechanisms for one-parameter
agents"): eine Zuteilung ist genau dann wahrheitsgetreu implementierbar, wenn die Arbeit $w(b)$ in der eigenen Meldung nicht steigt; die Zahlung
ist $b\,w(b) + \int_b^{\infty} w$. **Das Abschneiden des Integrals am Typbereich ($c_{hi} = 2$) ist unsere eigene, numerisch geprüfte
Anpassung** - ohne sie überzahlt die Formel auf dem beschränkten Bereich um ein Mehrfaches. Damit ist der Makespan-optimale wahrheitsgetreue
Mechanismus exakt bekannt (Lücke 0 %).

**Typ A - Bündelkosten-Vektor.** Der Typ eines Agenten sind seine $2^n-1$ Bündelkosten (aus einer zufälligen Instanz). Hier gibt es keine
solche Analytik; Fehlmeldung = jedes Bündelgebot mit einem eigenen Faktor $\lambda_S \in [0.5, 3]$ multiplizieren. Als wahrheitsgetreuer
Gegenspieler dient **Boosted VCG** (eigene Konstruktion, kein Zitat): VCG mit einem gebotsunabhängigen Bonus auf gleichmäßige Bündelgrößen.

**Das Netz.** Ein kleines tanh-MLP (numpy, Handbackprop) bildet das Meldungsprofil auf einen Softmax über *alle* $k^n$ Zuteilungen ab (XOR und
Zulässigkeit gelten per Konstruktion) und auf eine **Rente** $r_a$; Zahlung = erwartete gemeldete Kosten + softplus($r_a$) (ex-interim individuell
rational). Trainiert wird per Augmented Lagrangian: Makespan + $\mu\cdot$Rente + Regret-Nebenbedingung, der Regret wird durch Suche nach der besten
Fehlmeldung geschätzt (Typ B: Gitter, Typ A: Gradientenaufstieg). Der Regret-Preis $\kappa$ ist nach oben begrenzt - ohne diese Grenze kippt das
Training in einen degenerierten, Festpreis-artigen Zustand (Zuteilung fast konstant, riesige Rente - unsere Deutung).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
for row_start in range(0, len(preset_names), 4):
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, preset_names[row_start:row_start + 4]):
        with col:
            st.button(name, use_container_width=True, on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name))

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

# --- Seitenleiste --------------------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Einstellungen")
    kind = st.selectbox("Typ-Modell", options=["B", "A"], format_func=lambda key: KIND_LABELS[key], key="kind_select",
                        help="B: ein privater Kostenfaktor je Agent (exakte Analytik vorhanden). A: der ganze Bündelkosten-Vektor.")
    n_lo, n_hi = (2, 4) if kind == "B" else (3, 4)
    st.session_state["n_jobs_slider"] = min(max(int(st.session_state["n_jobs_slider"]), n_lo), n_hi)
    n_jobs = st.slider(
        "Anzahl Aufträge", n_lo, n_hi, key="n_jobs_slider",
        help="n = 2 (nur Typ B) ist das Handbeispiel: zwei gleiche Aufträge, keine Fahrzeit. Typ A: n = 4 nur als vorberechnete Wand.",
    )
    n_agents = st.slider("Anzahl Agenten", *bounds("n_agents_slider"), key="n_agents_slider")
    live_ok, precomputed = _supported(kind, int(n_jobs), int(n_agents))
    options = ([LIVE_LABEL] if live_ok else []) + precomputed
    if st.session_state["source_select"] == "live":
        st.session_state["source_select"] = LIVE_LABEL
    if st.session_state["source_select"] not in options:
        st.session_state["source_select"] = options[0] if options else None
    if options:
        source = st.selectbox(
            "Gelernter Mechanismus", options=options, key="source_select",
            format_func=lambda key: key if key == LIVE_LABEL else f"vorberechnet: {C.WEIGHT_SPECS[key][3]}",
            help="Live: jetzt in der App trainieren (nur Typ B mit höchstens 27 Zuteilungen). Vorberechnet: kleine Gewichte im Repo.",
        )
    else:
        source = None
        st.caption("Für diese Kombination gibt es keinen gelernten Mechanismus (weder live noch vorberechnet).")
    if source == LIVE_LABEL:
        kappa = st.slider("Regret-Preis κmax", *bounds("kappa_slider"), key="kappa_slider",
                          help="Obere Grenze des Regret-Preises: hoch = weniger Regret, aber schlechterer Makespan (Pareto-Knopf).")
        mu = st.slider("Rentengewicht μ", *bounds("mu_slider"), key="mu_slider",
                       help="Gewicht der Rente im Designer-Ziel: hoch = geringere Überzahlung.")
        steps = st.select_slider("Trainingsschritte", options=C.LIVE_STEPS_CHOICES, key="steps_select")
        learn_seed = st.number_input("Lern-Seed", *bounds("learn_seed_input"), key="learn_seed_input", step=1)
        st.button("🎲 Neuer Lern-Seed", use_container_width=True, on_click=randomize_seed)
    else:
        # Nur beim Live-Training wirksam: verborgen, die Werte bleiben erhalten.
        for key_ in ("kappa_slider", "mu_slider", "steps_select", "learn_seed_input"):
            st.session_state[key_] = st.session_state[key_]
        kappa, mu, steps, learn_seed = (st.session_state[k_] for k_ in ("kappa_slider", "mu_slider", "steps_select", "learn_seed_input"))
    gamma = st.slider("Faktor γ des Boosted VCG", *bounds("gamma_slider"), key="gamma_slider", step=0.5,
                      help="Bonus auf gleichmäßige Bündelgrößen: 0 = VCG; größer = besserer Makespan bei höherer Überzahlung, weiter Regret 0.")
    st.session_state["probe_agent_select"] = min(int(st.session_state["probe_agent_select"]), int(n_agents) - 1)
    probe = st.selectbox("Bieter der Misreport-Probe", options=list(range(int(n_agents))), format_func=lambda a: f"Agent {a + 1}",
                         key="probe_agent_select")

sync_query_params(kind, n_jobs, n_agents, source if source else "", kappa, mu, steps, learn_seed, gamma, probe)

kind, n_jobs, n_agents = str(kind), int(n_jobs), int(n_agents)
world = _world(kind, n_jobs, n_agents)
reports = _reports(kind, n_jobs, n_agents)

# --- Gelernten Mechanismus beschaffen -------------------------------------------------------------------

live_cfg = TrainConfig(hidden=QUICK_B.hidden, batch=QUICK_B.batch, steps=int(steps), mu=float(mu), kappa_max=float(kappa),
                       seed=int(learn_seed))
live_key = (kind, n_jobs, n_agents, live_cfg)
live_store = st.session_state.setdefault("live_results", {})
learned, history, train_seconds, train_note = None, None, None, ""
if source == LIVE_LABEL:
    result = live_store.get(live_key)
    if result is None and st.session_state.get("auto_train"):
        st.session_state["auto_train"] = False
        bar = st.progress(0.0, text="Trainiere den Mechanismus ...")

        def _progress(step, total, info):
            bar.progress(step / total, text=f"Schritt {step}/{total}: Lücke {info['makespan_gap']:.1f} %, Regret {info['regret']:.2f} min")

        result = train(world, live_cfg, progress=_progress)
        bar.empty()
        live_store[live_key] = result
    if result is not None:
        learned, history, train_seconds = result.mechanism, result.history, result.seconds
        train_note = "live trainiert"
elif source:
    learned, _, meta = _precomputed(source)
    history, train_seconds = meta.get("history"), meta.get("train_seconds")
    train_note = "vorberechnet"
mech_key = (source, live_key if source == LIVE_LABEL else None)

# --- 📐 Vergleich -----------------------------------------------------------------------------------------

st.subheader("📐 Vergleich: Handregeln, exakte Analytik und der gelernte Mechanismus")

if kind == "B" and n_jobs == 2:
    c_ex = np.array([[1.2, 1.0] + [1.0] * (n_agents - 2)])
    steps_ex = work_steps(world.S, c_ex[0], 0, 0.05, 4.0)
    pi_ex, pay_ex = threshold_mechanism(world, c_ex)
    p_ex = int(pi_ex.argmax(1)[0])
    cost_ex = c_ex[0, 0] * world.S[p_ex, 0]
    breaks = ", ".join(f"{end:.2f}" for _, end, _ in steps_ex[:-1])
    st.info(
        f"**Handbeispiel (2 Agenten, 2 gleiche Aufträge à 10 min, keine Fahrzeit):** Agent 1 hat den Kostenfaktor 1.2, Agent 2 den Faktor 1.0. "
        f"Die Makespan-optimale Zuteilung gibt jedem einen Auftrag (Makespan 12). Die Arbeit von Agent 1 fällt bei den Meldungen "
        f"**{breaks}** stufenweise ab ({' → '.join(f'{v:.0f}' for _, _, v in steps_ex)} min). Schwellenwert-Zahlung an Agent 1: "
        f"**{pay_ex[0, 0]:.1f}**, Kosten {cost_ex:.1f}, Nutzen **{pay_ex[0, 0] - cost_ex:.1f}** = das Integral der Arbeitskurve von 1.2 bis 2.0 "
        f"(10 · 0.8 = 8). Lügen bringt dort nichts: der Nutzen hängt nur vom Integral ab, nicht vom eigenen Gebot."
    )

if source is None:
    st.warning("Für diese Kombination gibt es keinen gelernten Mechanismus - gezeigt werden nur die Handregeln.")

if source == LIVE_LABEL and learned is None:
    st.info(
        "Noch nichts trainiert: der Vergleich zeigt zunächst Handregeln und exakte Analytik. Unten im Abschnitt **Training** startet ein Knopf "
        "das Live-Training (in der Seitenleiste einstellbar)."
    )

with st.spinner("Werte alle Mechanismen auf den festen Test-Stichproben aus ..."):
    rows = _comparison(kind, n_jobs, n_agents, mech_key, float(gamma), learned)
by_key = {r["key"]: r for r in rows}

cards = st.columns(4)
if "threshold" in by_key:
    thr = by_key["threshold"]
    cards[0].metric("Exakte Analytik", f"{thr['gap_pct']:.1f} % Lücke", help=f"Schwellenwert: Zahlung/Kosten {thr['pay_cost']:.2f}, Regret 0 (Theorie, numerisch geprüft).")
else:
    boost = by_key["boost"]
    cards[0].metric("Boosted VCG", f"{boost['gap_pct']:.1f} % Lücke", help=f"Zahlung/Kosten {boost['pay_cost']:.2f}, Regret 0 exakt.")
if "learned" in by_key:
    lr = by_key["learned"]
    cards[1].metric("Gelernt: Lücke", f"{lr['gap_pct']:.1f} %", help="Erwarteter Makespan über dem Optimum (randomisierte Zuteilung).")
    cards[2].metric("Gelernt: Zahlung / Kosten", f"{lr['pay_cost']:.2f}")
    cards[3].metric("Gelernt: Regret", f"{lr['regret']:.2f} min", help="Mittlerer Regret je Agent - nur eine Untergrenze.")
else:
    cards[1].metric("Gelernt: Lücke", "–")
    cards[2].metric("Gelernt: Zahlung / Kosten", "–")
    cards[3].metric("Gelernt: Regret", "–")

level, code, data = verdict(rows)
if code == "regret_high":
    st.warning(
        f"⚠️ **Der gelernte Mechanismus ist weit von wahrheitsgetreu**: Agenten gewinnen durch Fehlmeldung im Mittel {data['regret']:.2f} min "
        f"(Untergrenze). Lücke {data['gap']:.1f} %, Zahlung/Kosten {data['pay_cost']:.2f}. Mehr Regret-Druck (größeres κ, längeres Training) senkt "
        "den Regret, kostet aber Makespan."
    )
elif code == "dominated_by_truthful":
    st.warning(
        f"⚠️ **{data['label']} (exakt wahrheitsgetreu, Regret 0) ist auf beiden Achsen mindestens so gut**: {data['t_gap']:.1f} % Lücke bei "
        f"Zahlung/Kosten {data['t_pay']:.2f} - das gelernte Netz liegt bei {data['gap']:.1f} % / {data['pay_cost']:.2f} mit Regret {data['regret']:.2f} min."
    )
elif code == "learned_ok":
    st.success(f"✅ Gelernt: {data['gap']:.1f} % Lücke, Zahlung/Kosten {data['pay_cost']:.2f}, Regret {data['regret']:.2f} min (Untergrenze).")

note = rounding_note(rows)
if note is not None:
    st.info(
        f"ℹ️ **Nach dem Runden auf die wahrscheinlichste Zuteilung steigt der Regret von {note[0]:.2f} auf {note[1]:.2f} min**: die Wahrhaftigkeit gilt nur "
        "im Erwartungswert der randomisierten Zuteilung, nicht für eine ausgewürfelte."
    )

table = {
    "Mechanismus": [r["label"] for r in rows],
    "Lücke (Makespan)": [f"{r['gap_pct']:.1f} %" for r in rows],
    "Zahlung / Kosten": [f"{r['pay_cost']:.2f}" for r in rows],
    "IR verletzt": [f"{r['ir_share'] * 100:.0f} %" for r in rows],
    "Regret (min)": [f"{r['regret']:.2f}" for r in rows],
    "Agenten mit Regret > 0.1": [f"{r['regret_share'] * 100:.0f} %" for r in rows],
}
if kind == "A":
    table["Regret, nur uniformes λ-Gitter"] = [
        "–" if r.get("regret_grid") is None else f"{r['regret_grid']:.3f}" for r in rows
    ]
st.table(table)

family = []
for name, meta in _manifest().items():
    if (meta["kind"], meta["n"], meta["k"]) == (kind, n_jobs, n_agents):
        m = meta["metrics"]
        family.append({"label": meta["description"], "gap_pct": m["gap_pct"], "pay_cost": m["pay_cost"], "regret": m["regret"]})
st.plotly_chart(build_pareto(rows, family), use_container_width=True, key="pareto")
st.caption(
    "Jeder Punkt ein Mechanismus: links unten ist gut (kleine Lücke, wenig Überzahlung), die **Punktgröße ist der Regret** (klein = wahrheitsgetreu). "
    + ("Blau: die vorberechneten gelernten Varianten dieser Größe mit verschiedenem Regret-Preis. " if family else "")
    + "Der Regret ist eine **Untergrenze** - er ist so stark wie der Gegenspieler, der ihn sucht."
    + (" Die Zahlungen des exakten Schwellenwert-Mechanismus wurden auf den ersten 60 Test-Profilen ausgewertet." if kind == "B" else "")
)

st.markdown("---")

# --- Training --------------------------------------------------------------------------------------------

st.markdown("## 🏋️ Training")
if source == LIVE_LABEL:
    if learned is None:
        st.caption(
            f"Live-Training (numpy): {live_cfg.steps} Schritte, Netz {live_cfg.hidden}×{live_cfg.hidden}, Batch {live_cfg.batch}. "
            "Dauert je nach Größe wenige Sekunden bis etwa eine Viertelminute (Streamlit Cloud kann langsamer sein)."
        )
        if st.button("▶️ Trainieren", key="train_start"):
            st.session_state["auto_train"] = True
            st.rerun()
    else:
        st.caption(f"Live trainiert in {train_seconds:.1f} s.")
elif source:
    st.caption(f"Vorberechnet ({C.WEIGHT_SPECS[source][3]}), Training dauerte {train_seconds} s lokal. Live trainieren lässt sich nur Typ B mit höchstens 27 Zuteilungen.")
if history:
    st.plotly_chart(build_training_curves(history), use_container_width=True, key="training_curves")
    st.caption(
        "Makespan-Lücke, Rente und der im Training geschätzte Regret je Schritt (Mittel über 50 Schritte). Die ersten 150 Schritte laufen ohne Regret-Term; danach "
        "steigt der Regret-Preis. Der **Trainings-Regret unterschätzt** den Regret auf Test-Daten (der Gegenspieler im Training ist schwächer)."
    )

st.markdown("---")

# --- Misreport-Probe ---------------------------------------------------------------------------------------

st.markdown("## 🎭 Misreport-Probe: lohnt sich Lügen?")
probe = int(probe)
if kind == "B":
    pc1, pc2 = st.columns(2)
    with pc1:
        c_true = st.slider(f"Wahrer Kostenfaktor von Agent {probe + 1}", C_LO, C_HI, 1.2 if n_jobs == 2 else 1.0, step=0.05, key="probe_c_true")
    with pc2:
        c_other = st.slider("Kostenfaktor der anderen Agenten", C_LO, C_HI, 1.0, step=0.05, key="probe_c_other")
    profile = np.full((1, n_agents), c_other)
    profile[0, probe] = c_true
    grid = np.linspace(C_LO, C_HI, 120)
    batch = np.repeat(profile, len(grid), axis=0)
    batch[:, probe] = grid
    true_costs = world.costs(profile)[0][:, probe]

    def _utility_curve(mech):
        pi, pay = mech(world, batch)
        return pay[:, probe] - pi @ true_costs

    curves = [("A", "A: Pay-as-bid", _utility_curve(hand("A"))), ("B", "B: VCG", _utility_curve(hand("B"))),
              ("threshold", "exakte Schwelle", _utility_curve(lambda w, r: threshold_mechanism(w, r)))]
    if learned is not None:
        curves.append(("learned", "gelernt", _utility_curve(learned)))
        curves.append(("learned_rounded", "gelernt, gerundet", _utility_curve(learned.rounded_copy())))
    st.plotly_chart(build_misreport_b(grid, curves, c_true, probe), use_container_width=True, key="misreport_b")
    st.caption(
        "Nutzen von Agent %d gegen den Kostenfaktor, den er **meldet** (die anderen sind ehrlich). Ein wahrheitsgetreuer Mechanismus hat sein Maximum beim "
        "wahren Wert (gestrichelt); Pay-as-bid belohnt Aufblähen." % (probe + 1)
    )
    if n_agents == 2:
        st.markdown("**Allokationskarte:** erwartete Arbeit von Agent 1 über beide Kostenfaktoren")
        gg = np.linspace(C_LO, C_HI, 28)
        c1, c2 = np.meshgrid(gg, gg)
        cc = np.stack([c1.ravel(), c2.ravel()], axis=1)
        exact_pi, _ = hand_mechanism(world, cc, "A")
        exact_work = (exact_pi @ world.S[:, 0]).reshape(c1.shape)
        if learned is not None:
            logits, _, _ = learned.outputs(cc)
            from rn_nets import softmax
            learned_work = (softmax(logits) @ world.S[:, 0]).reshape(c1.shape)
        else:
            learned_work = np.full_like(exact_work, np.nan)
        st.plotly_chart(build_allocation_map(gg, learned_work, exact_work), use_container_width=True, key="alloc_map")
        st.caption(
            "Rechts die exakte Zuteilung: eine **Stufenfunktion** - je höher der eigene Kostenfaktor im Verhältnis zum anderen, desto weniger Arbeit "
            "(die Stufen liegen bei festen Verhältnissen der Faktoren). Links das gelernte Netz: die Stufen sind **verschmiert**, weil das Netz glatt ist "
            "und die Zuteilung randomisiert. Das kostet hier nur wenig Makespan (Tabelle oben); vermutlich ist es aber der Grund, warum der Regret nicht auf 0 fällt "
            "und warum das Runden ihn erhöht."
        )
else:
    lam_grid = np.array(LAMBDA_GRID)
    true_costs_all = world.costs(reports)

    def _lambda_curve(mech):
        vals = []
        for lam in lam_grid:
            rep = reports.copy()
            rep[:, probe, 1:] = reports[:, probe, 1:] * lam
            pi, pay = mech(world, rep)
            vals.append(float((pay[:, probe] - (pi * true_costs_all[:, :, probe]).sum(1)).mean()))
        return vals

    curves = [("A", "A: Pay-as-bid", _lambda_curve(hand("A"))), ("B", "B: VCG", _lambda_curve(hand("B"))),
              ("boost", f"Boosted VCG, γ = {gamma:g}", _lambda_curve(hand("boost", gamma)))]
    if learned is not None:
        curves.append(("learned", "gelernt", _lambda_curve(learned)))
    st.plotly_chart(build_misreport_a(lam_grid, curves, probe), use_container_width=True, key="misreport_a")
    st.caption(
        "Mittlerer Nutzen von Agent %d, wenn er **alle** seine Bündelgebote mit demselben Faktor λ skaliert (das gleichmäßige Gitter aus auction-demo). "
        "Beim gelernten Netz findet dieses Gitter fast nichts - das verteidigt das Training gezielt; die Gradientensuche über einzelne Bündel-Faktoren "
        "(Tabelle oben) findet deutlich mehr. **Ein kleiner Regret auf dem einen Gegenspieler ist keine Wahrhaftigkeit.**" % (probe + 1)
    )

st.markdown("---")

# --- Die Wand bei n = 4 -------------------------------------------------------------------------------------

st.markdown("## 🧱 Die Wand: warum n = 4 nicht mehr geht")
wall = _wall()
if wall is None:
    st.caption("Die vorberechnete Wand-Tabelle fehlt (`weights/wall.json`).")
elif st.session_state.get("wall_open"):
    st.plotly_chart(build_wall_chart(wall["plain"]), use_container_width=True, key="wall_chart")
    st.table({
        "n, k": [f"{r['n']}, {r['k']}" for r in wall["plain"]],
        "Zuteilungen k^n": [r["n"] and _fmt_int(r["k"] ** r["n"]) for r in wall["plain"]],
        "Lücke ohne jede Anreiz-Nebenbedingung": [f"{r['plain_gap']:.1f} %" for r in wall["plain"]],
        "Lücke gelernter Mechanismus (Typ A)": [("–" if r.get("learned_gap") is None else f"{r['learned_gap']:.1f} %") for r in wall["plain"]],
        "Regret gelernter Mechanismus (min)": [("–" if r.get("learned_regret") is None else f"{r['learned_regret']:.2f}") for r in wall["plain"]],
    })
    st.caption(
        "Das Netz muss unter $k^n$ Zuteilungen die beste wählen. Schon **ohne jeden Anreiz** (nur Makespan minimieren, 3000 Schritte, Netz 64×64, Typ-A-Instanzen) wird die "
        "Lücke ab n = 4 deutlich größer; mit Anreiz-Nebenbedingung kommt bei n = 4 ein großer Regret hinzu. Ein größeres Netz mit viel mehr Schritten senkt die Lücke "
        "(einmalige Prototyp-Messung für n = 4, k = 3: 11 % mit Netz 256×256 und 15000 Schritten), beseitigt sie aber nicht. Für n ≥ 5 müsste der Kopf faktorisiert werden (nicht gemessen)."
    )
else:
    st.button("🧱 Wand-Tabelle anzeigen (vorberechnet)", on_click=lambda: st.session_state.__setitem__("wall_open", True), key="wall_start")

st.markdown("---")

with st.expander("🧪 Was nicht funktioniert hat und wo die Grenzen liegen"):
    st.markdown(
        """
- **Kontext lernen (Typ B mit zufälliger Instanz):** der Kostenfaktor war privat, die Instanz zufällig und dem Netz als Eingabe gegeben. Das Netz lernte den
  Kontext nicht (Lücke um 18 %) - deshalb hat Typ B eine **feste** Instanz. (Einmalige Prototyp-Messung.)
- **Regret-Schätzer:** der Regret ist nur eine Untergrenze. Das Training unterschätzt ihn auf Test-Daten deutlich; das gleichmäßige λ-Gitter findet beim
  gelernten Netz fast nichts, die Gradientensuche mehr.
- **Rundung:** die Zuteilung des Netzes ist randomisiert. Rundet man auf die wahrscheinlichste Zuteilung, steigt der Regret stark - die Wahrhaftigkeit gilt nur im Erwartungswert.
- **Regret-Preis ohne Grenze:** ein unbegrenzt wachsender Regret-Preis (klassischer Augmented Lagrangian) kippt das Training in einen degenerierten,
  Festpreis-artigen Zustand (fast konstante Zuteilung, riesige Rente, Makespan +35 %, Regret ≈ 10 min - unsere Deutung). Deshalb ist κ hier gedeckelt.
- **Ex-interim-IR:** die Zahlung deckt die *erwarteten* Kosten der randomisierten Zuteilung, nicht jede ausgewürfelte.
- **Kein Beweis:** alles ist gemessen, nichts bewiesen; RegretNet ist ursprünglich verkaufsseitig (Erlösmaximierung), hier ist es Beschaffung mit Makespan-Ziel.
- **Größe:** Typ A nur bis n = 3, Typ B bis n = 4; der Softmax über alle $k^n$ Zuteilungen ist die Wand. Live-Laufzeiten auf Streamlit Cloud sind ungemessen.
        """
    )

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Zuteilung und Zahlung.** Netz $f_\theta(b) = (\text{logits}, r)$, $\pi = \text{softmax}(\text{logits})$ über alle $k^n$ Zuteilungen $p$,
$$\text{pay}_a(b) = \sum_p \pi_p(b)\, C^{\text{rep}}_{a}(p) + \text{softplus}(r_a(b)), \qquad
u_a(b'; v) = \text{pay}_a(b') - \sum_p \pi_p(b')\, C^{\text{true}}_{a}(p).$$

**Regret.** $\text{rgt}_a = \mathbb{E}_{v}\big[\max\big(0,\ \max_{b'_a} u_a(b'_a, v_{-a}; v_a) - u_a(v; v)\big)\big]$. Bei ehrlichem Bieten ist der Nutzen gleich der Rente.

**Training (Augmented Lagrangian, gedeckelt).**
$$\mathcal{L} = \mathbb{E}\Big[\sum_p \pi_p \max_a C_a(p)\Big] + \mu\,\mathbb{E}\Big[\sum_a \text{softplus}(r_a)\Big] + \sum_a \kappa_a\, \text{rgt}_a,
\quad \kappa_a = \min\big(\lambda_a + \rho\,\text{rgt}_a,\ \kappa_{\max}\big),$$
$\lambda_a \mathrel{+}= \rho\,\text{rgt}_a$ alle 40 Schritte, $\rho$ wächst um den Faktor 1.6. Die Fehlmeldung wird beim Ableiten als konstant behandelt
(Envelope-Gradient); der Gradient ist gegen finite Differenzen geprüft (Abweichung ~1e-8).

**Ein Parameter (Typ B).** Kosten $c_a W_a(S)$, Arbeit $w_a(b) = W_a(S^*(b, c_{-a}))$. Wahrheitsgetreu genau dann, wenn $w_a$ in $b$ nicht steigt; Zahlung
$P_a(b) = b\,w_a(b) + \int_b^{c_{hi}} w_a(u)\,du$ (gestutzt am Typbereich - eigene Anpassung), Nutzen bei Wahrheit $= \int_{c_a}^{c_{hi}} w_a(u)\,du \ge 0$.
Das Integral wird exakt berechnet: $w_a$ ist eine Stufenfunktion mit endlich vielen Sprungstellen.

**Boosted VCG (Typ A, eigene Konstruktion).** $S^* = \arg\min_S \sum_a C_a(S_a) + \gamma \sum_a |S_a|^2$, Zahlung
$p_a = h_{-a} - \big(f(S^*) - C_a(S^*_a)\big)$ mit $h_{-a}$ = Minimum von $f$ über Zuteilungen ohne Agent $a$. Exakt wahrheitsgetreu und individuell rational.

Implementiert in `rn_baselines.py`, `rn_mechanism.py`, `rn_train.py`, `rn_regret.py`, `rn_evaluation.py`.
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
