"""Defaults, Slider-Grenzen und Presets für die RegretNet-Demo.
Die Szenario-Konstanten (POSITION_RANGE_MAX ... SPIKE_MULTIPLIER) und die Gebotssprachen-Namen (von `auction_bids.py` gebraucht)
sind wortgleich aus auction-demo übernommen - dasselbe Vehikel; alles Übrige ist neu."""

# --- Vehikel (wortgleich aus auction-demo) -----------------------------------------------------------
POSITION_RANGE_MAX = 20.0
DURATION_BASE_RANGE = (5, 15)
SPIKE_PROBABILITY_SCALE = 0.4
SPIKE_MULTIPLIER = 4.0

LANG_ALL = "all"
LANG_SIZE = "size"
LANG_BLOCK1 = "block1"
LANG_BLOCK2 = "block2"

DEFAULT_SEED = 9


# --- Vorberechnete Mechanismen (weights/*.npz, erzeugt von rn_precompute.py) -------------------------
# Name -> (Typ, n, k, Beschreibung, TrainConfig-Überschreibungen). Typ B: Regret-Preis kappa_max ist der Pareto-Knopf.
WEIGHT_SPECS = {
    "b_n4k2_k03": ("B", 4, 2, "Regret-Preis κ = 0.3", dict(hidden=64, batch=256, steps=5000, kappa_max=0.3)),
    "b_n4k2_k1": ("B", 4, 2, "Regret-Preis κ = 1", dict(hidden=64, batch=256, steps=5000, kappa_max=1.0)),
    "b_n4k2_k3": ("B", 4, 2, "Regret-Preis κ = 3", dict(hidden=64, batch=256, steps=5000, kappa_max=3.0)),
    "b_n4k2_k10": ("B", 4, 2, "Regret-Preis κ = 10", dict(hidden=64, batch=256, steps=5000, kappa_max=10.0)),
    "b_n3k2_k3": ("B", 3, 2, "Regret-Preis κ = 3", dict(hidden=64, batch=256, steps=4000, kappa_max=3.0)),
    "b_n4k3_k3": ("B", 4, 3, "Regret-Preis κ = 3", dict(hidden=96, batch=256, steps=4000, kappa_max=3.0)),
    "a_n3k2_hi": ("A", 3, 2, "wenig Regret-Druck (κ ≤ 30)", dict(steps=4000, kappa_max=30.0, lr0=0.005, lr1=0.0005)),
    "a_n3k2_mid": ("A", 3, 2, "mittlerer Regret-Druck (κ ≤ 100)", dict(steps=4000, kappa_max=100.0, lr0=0.005, lr1=0.0005)),
    "a_n3k2_low": ("A", 3, 2, "hoher Regret-Druck (κ ≤ 100, stärkere Suche)", dict(steps=4000, kappa_max=100.0, lr0=0.01, lr1=0.001, ascent_steps=16)),
    "a_n3k2_zero": ("A", 3, 2, "höchster Regret-Druck (κ ≤ 100, kleine Lernrate)", dict(steps=4000, kappa_max=100.0, lr0=0.002, lr1=0.0002)),
    "a_n3k3_mid": ("A", 3, 3, "mittlerer Regret-Druck (κ ≤ 100)", dict(steps=3000, kappa_max=100.0, lr0=0.005, lr1=0.0005)),
    "a_n4k2_mid": ("A", 4, 2, "n = 4: die Wand (κ ≤ 100)", dict(steps=3000, kappa_max=100.0, lr0=0.005, lr1=0.0005)),
}


# --- Regler ---------------------------------------------------------------------------------------
KAPPA_MIN, KAPPA_MAX = 0.3, 30.0        # Regret-Preis κmax des Live-Trainings (Pareto-Knopf)
MU_MIN, MU_MAX = 0.02, 0.5              # Gewicht der Rente im Designer-Ziel
LIVE_STEPS_CHOICES = (1000, 2000, 2500, 4000)
LIVE_MAX_PARTITIONS = 27                # Live-Training nur bis k^n = 27 (schneller Kopf); größer: nur vorberechnet
N_TEST_B, N_TEST_A = 200, 60            # feste Test-Stichproben der App (Seeds ab 100000)

# Presets: kind (A/B), n, k, source ("live" oder Name in WEIGHT_SPECS), Live-Regler, gamma des Boosted VCG, Bieter der Probe (0-basiert)
_P = {"kappa": 3.0, "mu": 0.1, "steps": 2500, "lseed": 0, "gamma": 3.0, "probe": 0}
PRESETS = {
    "Handbeispiel 2×2": {**_P, "kind": "B", "n": 2, "k": 2, "source": "live", "steps": 1000},
    "Ein Parameter: Analytik gewinnt": {**_P, "kind": "B", "n": 4, "k": 2, "source": "b_n4k2_k3"},
    "Pay-as-bid lügt": {**_P, "kind": "A", "n": 3, "k": 2, "source": "a_n3k2_mid"},
    "VCG: wahrheitsgetreu, aber Lücke": {**_P, "kind": "A", "n": 3, "k": 2, "source": "a_n3k2_mid", "probe": 1},
    "Boosted VCG": {**_P, "kind": "A", "n": 3, "k": 2, "source": "a_n3k2_hi", "gamma": 3.0},
    "Regret gegen Überzahlung": {**_P, "kind": "B", "n": 4, "k": 2, "source": "b_n4k2_k03"},
    "Rundung zerstört Wahrhaftigkeit": {**_P, "kind": "B", "n": 4, "k": 2, "source": "b_n4k2_k10"},
    "Die Wand bei n = 4": {**_P, "kind": "A", "n": 4, "k": 2, "source": "a_n4k2_mid"},
}
PRESET_HELP = {
    "Handbeispiel 2×2": "Zwei gleiche Aufträge, zwei Agenten: die exakte Schwellenwert-Analytik ist von Hand herleitbar (Sprungstellen 0.5 und 2.0, "
        "Zahlung 20, Nutzen 8). Das Netz wird live trainiert und liegt nach wenigen Sekunden nah dran - mit kleinem Restregret.",
    "Ein Parameter: Analytik gewinnt": "Ein privater Kostenfaktor je Agent: die exakte Schwellenwert-Analytik hat 0 % Lücke, Regret 0 und etwas weniger "
        "Überzahlung als das gelernte Netz, das nur nahe herankommt.",
    "Pay-as-bid lügt": "Bündelkosten-Vektoren: Pay-as-bid auf der Makespan-optimalen Zuteilung belohnt Aufblähen - der Regret liegt bei rund 27 Minuten. "
        "Die Misreport-Probe zeigt, wo sich Lügen lohnt.",
    "VCG: wahrheitsgetreu, aber Lücke": "VCG hat Regret 0 - aber es minimiert die Gesamtzeit und lässt den Makespan um etwa 15 % über dem Optimum. "
        "Das gelernte Netz ist bei mehr als einer Minute Regret.",
    "Boosted VCG": "Ein gebotsunabhängiger Bonus auf gleichmäßige Bündelgrößen (eigene Konstruktion) senkt die Lücke gegenüber VCG bei Regret 0 - "
        "und kostet Überzahlung. Das gelernte Netz mit wenig Regret-Druck liegt bei mehreren Minuten Regret.",
    "Regret gegen Überzahlung": "Mit sehr wenig Regret-Druck (κ = 0.3) zahlt das Netz weniger als die exakte Analytik, hat dafür aber einen Regret von "
        "mehreren Minuten. Die blauen Punkte im Diagramm zeigen die ganze Reihe: mehr Regret-Druck senkt den Regret und erhöht die Zahlungen.",
    "Rundung zerstört Wahrhaftigkeit": "Das Netz hat einen kleinen Regret, solange die Zuteilung randomisiert bleibt. Rundet man auf die wahrscheinlichste "
        "Zuteilung, steigt der Regret um ein Vielfaches: die Wahrhaftigkeit gilt nur im Erwartungswert.",
    "Die Wand bei n = 4": "Bei vier Aufträgen wählt das Netz unter 16 Zuteilungen - schon ohne jede Anreiz-Nebenbedingung wird die Lücke groß, mit ihr kommen "
        "mehrere Minuten Regret dazu. Die Wand-Tabelle unten zeigt es für mehrere Größen.",
}


def _band(value, tolerance):
    return (value - tolerance, value + tolerance)


# Erwartete Messwerte je Preset (mit dem ausgelieferten Code kalibriert; tests/test_presets.py prüft sie). Schlüssel: Kennzahlen aus
# `comparison_table` (learned_*, rounded_*, A_*, B_*, boost_*, threshold_*), Text = erwarteter Verdict-Code, Tupel = (min, max).
PRESET_EXPECTED_BANDS = {
    "Handbeispiel 2×2": {"verdict": "dominated_by_truthful", "learned_gap": (0.0, 3.0), "learned_regret": (0.0, 0.15), "rounded_regret": (0.2, 0.7),
                         "threshold_pay": (1.44, 1.56)},
    "Ein Parameter: Analytik gewinnt": {"verdict": "dominated_by_truthful", "learned_gap": (0.1, 3.1), "learned_regret": (0.03, 0.25),
                                        "learned_pay": (1.50, 1.62), "threshold_pay": (1.45, 1.55)},
    "Pay-as-bid lügt": {"verdict": "regret_high", "A_regret": (24.0, 31.0), "A_gap": (-0.1, 0.5)},
    "VCG: wahrheitsgetreu, aber Lücke": {"B_gap": (12.0, 18.5), "B_regret": (-1e-9, 1e-9), "B_pay": (1.2, 1.35)},
    "Boosted VCG": {"boost_gap": (7.0, 13.0), "boost_pay": (1.64, 1.76), "B_gap": (12.0, 18.5), "learned_regret": (2.0, 5.0)},
    "Regret gegen Überzahlung": {"verdict": "regret_high", "learned_regret": (2.0, 5.0), "learned_pay": (1.41, 1.51)},
    "Rundung zerstört Wahrhaftigkeit": {"learned_regret": (0.0, 0.15), "rounded_regret": (0.2, 0.6)},
    "Die Wand bei n = 4": {"verdict": "regret_high", "learned_regret": (2.0, 4.5), "learned_gap": (22.0, 30.0), "B_gap": (19.0, 27.0)},
}
