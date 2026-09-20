# RegretNet: ein gelernter Mechanismus für die Kran-Auftragsvergabe – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-regretnet-demo.streamlit.app/)**

Achtes und **letztes** Stück der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations
Research und Machine Learning", **Multi-Agenten-Koordinations-Linie** - die **Fortsetzung von** [auction-demo](../auction-demo)
(Wurzel der Linie: [contract-net-demo](../contract-net-demo)). Das Vehikel (Kran-Aufträge auf einer 1-D-Schiene, Makespan-Ziel) und die
Bündelgebote sind wortgleich aus auction-demo übernommen; die Handregeln von dort sind die Vergleichsschicht.

> **Ein teils negatives Ergebnis, ausdrücklich so gemeint.** Gemessen: bei **einem** privaten Parameter je Agent gibt es eine exakt
> wahrheitsgetreue Analytik, die das gelernte Netz auf beiden Achsen (Lücke, Überzahlung) und beim Regret schlägt; bei
> **Bündelkosten-Vektoren** approximiert das Netz nur wahrheitsgetreue Varianten von VCG und bricht ab n = 4 ein. Der Regret ist nur
> eine **Untergrenze**, die Zuteilung des Netzes ist **randomisiert**, und nach dem Runden steigt der Regret um ein Vielfaches. Nichts
> davon ist bewiesen; alles ist gemessen. Zitiert wird nur: Dütting, Feng, Narasimhan, Parkes, Ravindranath, *Optimal Auctions through
> Deep Learning*, ICML 2019 (RegretNet); Archer & Tardos, *Truthful mechanisms for one-parameter agents*, FOCS 2001 (mit dem Vorbehalt
> unten) und Gerkey & Matarić 2004 (Auktionen in der Multi-Roboter-Aufgabenverteilung, siehe auction-demo).

## Was dieses Stück tut

auction-demo zeigte: keine handentworfene Zahlungsregel erfüllt gleichzeitig Makespan-Optimum, Kostendeckung und Wahrhaftigkeit. Hier lernt
ein Netz **Zuteilung und Zahlung** und wird gegen die exakte Analytik gemessen.

**Setting (Beschaffung, wie in auction-demo).** Der Auktionator kauft die Bearbeitung; die Kosten eines Agenten sind seine
Fertigstellungszeit, 1 Geldeinheit = 1 Minute; dass er lügen könnte, ist Modellierung. Der **Regret** eines Agenten ist der größte Nutzengewinn
durch eine Fehlmeldung bei ehrlichen anderen (Minuten, Mittel über Test-Instanzen).

**Das Netz.** Ein tanh-MLP (numpy, Handbackprop) bildet das Meldungsprofil auf einen **Softmax über alle k^n Zuteilungen** (XOR und Zulässigkeit
per Konstruktion) und eine **Rente** r_a ab. Zahlung = erwartete gemeldete Kosten + softplus(r_a) (ex-interim individuell rational bezüglich der
Meldungen; ex-post-IR für eine ausgewürfelte Zuteilung gilt *nicht*). Training per **Augmented Lagrangian**: erwarteter Makespan + μ·Rente +
Regret-Nebenbedingung; der Regret wird durch Suche nach der besten Fehlmeldung geschätzt (Typ B: 1-D-Gitter, Typ A: normierter Gradientenaufstieg mit
Restarts), die Fehlmeldung beim Ableiten als konstant behandelt (Envelope-Gradient). Der Gradient ist gegen finite Differenzen geprüft (~1e-8).
**Der Regret-Preis κ ist nach oben begrenzt**: mit dem klassisch unbegrenzt wachsenden Multiplikator kippte das Training (Typ B, n = 4, k = 2, 2000 Schritte)
in einen Zustand mit fast konstanter Zuteilung, sehr großer Rente (~29 min), Makespan +35 % und Regret ≈ 10 min - ein Festpreis-artiger Mechanismus (Deutung, nicht
separat verifiziert). Der Multiplikator wächst früh an dem großen Anfangsregret und dominiert dann alles. Auch die Lernrate war entscheidend: mit 2e-3 war das
Training bei gleicher Schrittzahl deutlich schlechter als mit 2e-2 (einmalige Beobachtung; Standard ist 2e-2 fallend auf 1e-3).

### Typ B – ein Parameter (Hauptstück)

Feste Instanz; jeder Agent kennt einen privaten Kostenfaktor c_a ~ U[0.5, 2], seine Kosten sind c_a · W_a(S) (öffentliche Bündelzeit). Dann gilt
nach Archer & Tardos: eine Zuteilung ist genau dann wahrheitsgetreu implementierbar, wenn die Arbeit w(b) in der eigenen Meldung nicht steigt; die Zahlung
ist b·w(b) + ∫_b^∞ w. **Vorbehalt:** die Zitierung stützt sich auf mein Gedächtnis (etwa 95 % sicher, dass der Satz so dort steht), nicht auf ein
nachgeschlagenes Exemplar. **Das Abschneiden des Integrals am Typbereich c_hi = 2 ist unsere eigene, numerisch geprüfte Anpassung**: ohne sie überzahlt
die Formel auf dem beschränkten Bereich um ein Mehrfaches. Damit ist der Makespan-optimale wahrheitsgetreue Mechanismus exakt bekannt.

**Handbeispiel (Preset, von Hand herleitbar):** zwei gleiche Aufträge à 10 min, keine Fahrzeit, c₁ = 1.2, c₂ = 1.0. Die Arbeit von Agent 1 ist 20 für
Meldungen unter 0.5, 10 bis 2.0, danach 0; Schwellenwert-Zahlung 20, Kosten 12, Nutzen 8 = 10·(2.0 − 1.2). Ein live trainiertes Netz (3 s) erreicht
0.8 % Lücke, Zahlung/Kosten 1.57 (Schwelle 1.49), Regret 0.02 min - und 0.39 min nach dem Runden.

Fahrzeug-Instanz, n = 4, k = 2, feste Test-Stichprobe (200 Profile, Seeds ab 100000):

| Mechanismus | Lücke | Zahlung/Kosten | Regret (min) |
|---|---|---|---|
| Schwellenwert (exakt) | 0.0 % | 1.50 | 0 |
| A: Makespan + Pay-as-bid | 0.0 % | 1.00 | 41.8 |
| B: Summe + VCG | 51.9 % | 1.55 | 0 |
| E: Makespan + Clarke-Formel | 0.0 % | 1.11 (IR verletzt bei 57 %) | 9.6 |
| gelernt, κ ≤ 0.3 / 1 / 3 / 10 | 1.7 / 1.8 / 1.6 / 1.8 % | 1.46 / 1.52 / 1.56 / 1.57 | 3.53 / 1.27 / 0.105 / 0.043 |
| gelernt, gerundet (dieselben) | 1.8 / 1.8 / 1.6 / 1.6 % | 1.46 / 1.52 / 1.56 / 1.57 | 4.11 / 1.68 / 0.41 / 0.38 |

Der Regret-Preis handelt Regret gegen **Überzahlung**, kaum gegen Makespan; das gelernte Netz liegt bei 1.6-1.8 % Lücke, während die exakte Analytik
0 % bei geringerer Zahlung und Regret 0 hat. Fehlmeldungen sucht ein feines 1-D-Gitter (200 Punkte) - in einem Parameter praktisch exakt.

### Typ A – Bündelkosten-Vektor (sekundär)

Der Typ eines Agenten sind seine 2^n − 1 Bündelkosten (aus einer zufälligen Instanz); Fehlmeldung = jedes Bündelgebot mit einem eigenen Faktor
λ_S ∈ [0.5, 3]. Hier gibt es keine bekannte optimale wahrheitsgetreue Analytik. Als wahrheitsgetreuer Gegenspieler dient **Boosted VCG (eigene
Konstruktion, kein Zitat)**: Summenziel + γ·Σ|Bündel|² (gebotsunabhängiger Bonus auf gleichmäßige Bündelgrößen) mit der Clarke-Zahlung des Zielwerts -
exakt wahrheitsgetreu und individuell rational (Regret 0 gemessen; γ = 0 ist VCG).

n = 3, k = 2, 60 Test-Instanzen (`weights/`, vorberechnet):

| Mechanismus | Lücke | Zahlung/Kosten | Regret (min) |
|---|---|---|---|
| A: Pay-as-bid | 0.0 % | 1.00 | 27.4 |
| B: VCG | 15.2 % | 1.27 | 0 |
| Boosted VCG, γ = 3 | 10.0 % | 1.70 | 0 |
| E: Clarke auf Makespan | 0.0 % | 1.10 (IR verletzt bei 39 %) | 3.9 |
| gelernt, wenig → höchster Regret-Druck | 5.1 / 11.5 / 19.8 / 27.1 % | 1.30 / 1.32 / 1.32 / 1.34 | 3.42 / 1.77 / 0.61 / 0.18 |
| gelernt, gerundet (dieselben) | 1.6 / 3.9 / 6.1 / 17.9 % | ≈ 1.3 | 2.1 / 4.2 / 6.2 / 15.0 |

Bei Regret ≲ 1 liegt das Netz **hinter** VCG (Lücke 19.8 % gegen 15.2 %); nur bei mehreren Minuten Regret schlägt es die Lücke von VCG. Der Regret hängt
stark vom Gegenspieler ab: das gleichmäßige λ-Gitter aus auction-demo findet beim gelernten Netz fast nichts (0.001-0.03 min), die Gradientensuche über
einzelne Bündel-Faktoren deutlich mehr - genau das Gitter verteidigt das Training gezielt.

### Die Wand bei n = 4

Der Softmax über alle k^n Zuteilungen ist der Engpass. Schon **ohne jede Anreiz-Nebenbedingung** (nur Makespan minimieren, 3000 Schritte, Netz 64×64,
Typ-A-Instanzen) liegt die Lücke bei 0.4 % (n=3, k=2), 9.2 % (3, 3), 19.7 % (4, 2) und 19.3 % (4, 3); der gelernte Mechanismus bei n = 4, k = 2 hat 26.5 % Lücke und 3.0 min
Regret (17.4 nach Runden). Ein größeres Netz mit deutlich mehr Schritten senkt die Lücke (einmalige Prototyp-Messung für n = 4, k = 3: 11 % mit Netz
256×256 und 15000 Schritten), beseitigt sie aber nicht. Für n ≥ 5 müsste der Kopf faktorisiert werden (nicht gemessen).

## Was nicht funktioniert hat / Grenzen

- **Kontext lernen (Typ B mit zufälliger Instanz):** der Kostenfaktor privat, die Instanz zufällig und dem Netz als Eingabe gegeben - das Netz lernte den Kontext
  nicht (Lücke um 18 %). Deshalb hat Typ B eine **feste** Instanz. (Einmalige Prototyp-Messung.)
- **Unbegrenzter Regret-Preis** (klassischer Augmented Lagrangian): Kollaps in einen Festpreis-artigen Zustand (Deutung), siehe oben.
- **Regret nur als Untergrenze;** der Trainings-Regret unterschätzt den Test-Regret deutlich (der Gegenspieler im Training ist schwächer).
- **Randomisierte Zuteilung:** die Wahrhaftigkeit gilt nur im Erwartungswert; Runden erhöht den Regret - relativ umso stärker, je kleiner er vorher war: Typ B von ×1.2 (κ ≤ 0.3) über ×4 (κ ≤ 3) bis ×9 (κ ≤ 10), im Handbeispiel ×18, in Typ A bis ×85.
- **Ex-interim-IR** statt ex-post-IR; **kein Beweis**; RegretNet ist ursprünglich verkaufsseitig (Erlösmaximierung), hier ist es Beschaffung mit Makespan-Ziel.
- **Größe:** Typ A nur bis n = 3 (n = 4 nur als vorberechnete Wand), Typ B bis n = 4; Live-Laufzeiten auf Streamlit Cloud sind ungemessen (lokal: Handbeispiel 3 s,
  n = 4 / k = 2 etwa 10 s).
- Typbereich [0.5, 2] gewählt - die Rente der exakten Analytik (~50 % der Kosten) hängt daran.

## Verifikation

- Gradienten des Surrogat-Verlusts und der Eingabe-Gradient des Netzes gegen finite Differenzen (~1e-8 / 1e-9), Typ A und B.
- Handregeln gegen `auction_mechanism.run_mechanism` aus auction-demo (Zuteilung und Zahlungen); VCG und Boosted VCG: Regret 0 (Gitter und Zufalls-Hill-Climbing), IR.
- Handbeispiel (Sprungstellen 0.5 und 2.0, Zahlung 20, Nutzen 8); Schwellenwert-Mechanismus wahrheitsgetreu und IR auf einer Stichprobe, Arbeitskurve monoton.
- Netz: Softmax-Summe 1, Nutzen bei ehrlichem Bieten = Rente > 0, Speichern/Laden, deterministisches Training, Live-Training auf dem Handbeispiel in Qualitätsband.
- Alle 8 Presets in ihren kalibrierten Bändern (`tests/test_presets.py`); AppTest-Rauchtests (Standard, jedes Preset, Live-Training, verborgene Regler).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Vergleich, Training, Misreport-Probe, Allokationskarte, Wand |
| `rn_types.py`, `rn_partitions.py` | Welten (Typ B / A), alle Zuteilungen, vektorisierter Generator |
| `rn_baselines.py` | Handregeln A/B/E, Boosted VCG, exakte gestutzte Schwelle |
| `rn_nets.py`, `rn_mechanism.py`, `rn_train.py` | tanh-MLP mit Eingabe-Gradient, gelernter Mechanismus, Augmented-Lagrangian-Training |
| `rn_regret.py`, `rn_evaluation.py` | Regret-Schätzer, Kennzahlen, Vergleichstabelle, Verdict |
| `rn_precompute.py`, `weights/` | Vorberechnung (`python rn_precompute.py`), kleine npz-Gewichte, Manifest, Wand-Tabelle |
| `auction_*.py`, `cn_*.py` | Vehikel und Bündelgebote (wortgleich aus auction-demo; `cn_visualization.py` mit `lock_axes`) |
| `tests/` | Kern, Presets, AppTest, Vehikel |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
