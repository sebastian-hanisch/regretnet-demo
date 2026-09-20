"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem OR-Demo-Portfolio, siehe
constraint-programming-demo/csp_presets.py). Regler: Typ-Modell, Aufträge, Agenten, gelernter Mechanismus (Live oder vorberechnet),
Live-Training (Regret-Preis, Rentengewicht, Schritte, Lern-Seed), Faktor γ des Boosted VCG und der Bieter der Misreport-Probe."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import cn_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "kind_select": SettingSpec("kind", str, "B"),
    "n_jobs_slider": SettingSpec("n", int, 4, 2, 4),
    "n_agents_slider": SettingSpec("k", int, 2, 2, 3),
    "source_select": SettingSpec("src", str, "b_n4k2_k3"),
    "kappa_slider": SettingSpec("kappa", float, 3.0, C.KAPPA_MIN, C.KAPPA_MAX),
    "mu_slider": SettingSpec("mu", float, 0.1, C.MU_MIN, C.MU_MAX),
    "steps_select": SettingSpec("steps", int, 2500, min(C.LIVE_STEPS_CHOICES), max(C.LIVE_STEPS_CHOICES)),
    "learn_seed_input": SettingSpec("lseed", int, 0, 0, 2_000_000_000),
    "gamma_slider": SettingSpec("gamma", float, 3.0, 0.0, 8.0),
    "probe_agent_select": SettingSpec("probe", int, 0, 0, 2),
}


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default
    for flag in ("auto_train",):
        if flag not in st.session_state:
            st.session_state[flag] = False


def snap_steps(value):
    return min(C.LIVE_STEPS_CHOICES, key=lambda choice: abs(choice - value))


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    if st.session_state.get("kind_select") not in ("A", "B"):
        st.session_state["kind_select"] = "B"
    st.session_state["steps_select"] = snap_steps(st.session_state.get("steps_select", 2500))
    st.session_state["permalink_loaded"] = True


def sync_query_params(kind, n_jobs, n_agents, source, kappa, mu, steps, learn_seed, gamma, probe):
    try:
        st.query_params["kind"] = str(kind)
        st.query_params["n"] = str(int(n_jobs))
        st.query_params["k"] = str(int(n_agents))
        st.query_params["src"] = str(source)
        st.query_params["kappa"] = str(kappa)
        st.query_params["mu"] = str(mu)
        st.query_params["steps"] = str(int(steps))
        st.query_params["lseed"] = str(int(learn_seed))
        st.query_params["gamma"] = str(gamma)
        st.query_params["probe"] = str(int(probe))
    except Exception:
        pass


def apply_preset(name):
    p = C.PRESETS[name]
    st.session_state["kind_select"] = p["kind"]
    st.session_state["n_jobs_slider"] = p["n"]
    st.session_state["n_agents_slider"] = p["k"]
    st.session_state["source_select"] = p["source"]
    st.session_state["kappa_slider"] = p["kappa"]
    st.session_state["mu_slider"] = p["mu"]
    st.session_state["steps_select"] = p["steps"]
    st.session_state["learn_seed_input"] = p["lseed"]
    st.session_state["gamma_slider"] = p["gamma"]
    st.session_state["probe_agent_select"] = p["probe"]
    st.session_state["auto_train"] = p["source"] == "live"


def randomize_seed():
    st.session_state["learn_seed_input"] = random.randint(0, 2_000_000_000)
