"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Randgrößen, Live-Training, verborgene Regler."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import cn_constants as C

APP = Path(__file__).resolve().parent.parent / "app.py"


def _run(setup=None, timeout=120):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    # Zweistufig: erst Typ/n/k setzen und rechnen (die Optionen des Mechanismus-Reglers hängen davon ab), dann der Rest.
    # (Im echten Betrieb setzt der Preset-Knopf alles in einem Callback vor dem Rerun - nur AppTest braucht die zwei Läufe.)
    at.session_state["kind_select"] = p["kind"]
    at.session_state["n_jobs_slider"] = p["n"]
    at.session_state["n_agents_slider"] = p["k"]
    at.session_state["source_select"] = None
    at.run()
    at.session_state["source_select"] = "Live trainieren" if p["source"] == "live" else p["source"]
    at.session_state["kappa_slider"] = p["kappa"]
    at.session_state["mu_slider"] = p["mu"]
    at.session_state["steps_select"] = p["steps"]
    at.session_state["gamma_slider"] = p["gamma"]
    at.session_state["probe_agent_select"] = p["probe"]
    at.session_state["auto_train"] = p["source"] == "live"


def test_default_renders_without_exception():
    at = _run()
    assert any("Vergleich" in h.value for h in at.subheader)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders(name):
    _run(lambda at: _apply(at, C.PRESETS[name]))


def test_live_training_button_flow_hand_example():
    at = _run(lambda at: _apply(at, C.PRESETS["Handbeispiel 2×2"]))
    assert not any("Noch nichts trainiert" in i.value for i in at.info)            # auto_train hat trainiert


def test_hidden_live_controls_keep_their_state_and_dependent_bounds_are_clamped():
    def setup(at):
        at.session_state["kappa_slider"] = 7.0
        at.session_state["n_jobs_slider"] = 4
    at = _run(setup)
    assert not any(s.key == "kappa_slider" for s in at.slider)                    # vorberechnet: Live-Regler verborgen
    assert at.session_state["kappa_slider"] == 7.0                                 # Wert bleibt erhalten
    at.session_state["kind_select"] = "A"
    at.session_state["n_jobs_slider"] = 2                                          # Typ A kennt n = 2 nicht
    at.run()
    assert not at.exception
    assert at.session_state["n_jobs_slider"] >= 3


def test_combinations_without_any_mechanism_render():
    def setup(at):
        at.session_state["kind_select"] = "A"
        at.session_state["n_jobs_slider"] = 4
        at.session_state["n_agents_slider"] = 3                                   # weder live noch vorberechnet
    _run(setup)
