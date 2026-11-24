"""
scenarios.py
============
Synthetic score-vector generators for the two Monte Carlo suites.

Every generator has the signature  g(rng) -> np.ndarray  and consumes random
numbers from the numpy Generator `rng` in a fixed order (documented in each
SPEC entry). Reproducing the paper therefore needs only: the generator
definitions below, numpy's default_rng (PCG64), the seed, the trial count and
the order in which scenarios are drawn (the order of the dicts below).

Notation used in SPEC (and in the generated LaTeX tables):
  U(a,b)   uniform on [a,b];   N(m,sd) normal;   Int[a,b] integer uniform, inclusive
  Pareto(3) numpy's Lomax/Pareto-II with shape 3;   Beta(a,b) beta distribution
  clip[a,b] elementwise clipping

MAIN SUITE (8 scenarios): general distributional stress test. Individual scores
can exceed theta_m, so all three decisions can occur.

SUB-THRESHOLD SUITE (9 scenarios): every score is clipped to SIGNAL_CAP = 0.68 <
theta_m = 0.70, so no individual score can trigger Remove; isolates what the
pooling and adjustment stages do on their own.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Main suite (signal counts 2..12 unless stated)
# ---------------------------------------------------------------------------
MIN_SIGNALS, MAX_SIGNALS = 2, 12


def _n(rng, lo=MIN_SIGNALS, hi=MAX_SIGNALS):
    return int(rng.integers(lo, hi + 1))


def g_many_weak(rng):
    n = _n(rng, 6, MAX_SIGNALS)
    return rng.uniform(0.05, 0.25, size=n)


def g_one_strong_many_weak(rng):
    n_weak = _n(rng, 4, MAX_SIGNALS - 1)
    weak = rng.uniform(0.05, 0.25, size=n_weak)
    strong = rng.uniform(0.75, 0.95, size=1)
    return np.concatenate([weak, strong])


def g_medium_corroboration(rng):
    n = _n(rng)
    center = rng.uniform(0.35, 0.55)
    return np.clip(center + rng.normal(0, 0.05, size=n), 0.0, 1.0)


def g_correlated(rng):
    n = _n(rng)
    base = rng.uniform(0.1, 0.9)
    return np.clip(base + rng.normal(0, 0.03, size=n), 0.0, 1.0)


def g_duplicate(rng):
    n = _n(rng, 5, MAX_SIGNALS)
    return np.full(n, rng.uniform(0.1, 0.4))


def g_heavy_tailed(rng):
    n = _n(rng)
    return np.clip(rng.pareto(a=3.0, size=n) * 0.15, 0.0, 1.0)


def g_uniform(rng):
    n = _n(rng)
    return rng.uniform(0.0, 1.0, size=n)


def g_random_beta(rng):
    n = _n(rng)
    a = rng.uniform(0.5, 5.0)
    b = rng.uniform(0.5, 5.0)
    return rng.beta(a, b, size=n)


MAIN_SUITE = {
    "ManyWeak": g_many_weak,
    "OneStrongManyWeak": g_one_strong_many_weak,
    "MediumCorroboration": g_medium_corroboration,
    "Correlated": g_correlated,
    "Duplicate": g_duplicate,
    "HeavyTailed": g_heavy_tailed,
    "Uniform": g_uniform,
    "RandomBeta": g_random_beta,
}

MAIN_SPEC = {
    "ManyWeak": ("Int[6,12]", "$s_i\\sim U(0.05,0.25)$ i.i.d.",
                 "many weak scores; no score reaches $\\theta_r$"),
    "OneStrongManyWeak": ("Int[4,11]+1", "weak $s_i\\sim U(0.05,0.25)$; one strong $s^*\\sim U(0.75,0.95)$",
                          "one decisive score among weak ones"),
    "MediumCorroboration": ("Int[2,12]", "$c\\sim U(0.35,0.55)$; $s_i=\\mathrm{clip}_{[0,1]}(c+\\epsilon_i)$, $\\epsilon_i\\sim N(0,0.05)$",
                            "moderate, clustered scores"),
    "Correlated": ("Int[2,12]", "$b\\sim U(0.1,0.9)$; $s_i=\\mathrm{clip}_{[0,1]}(b+\\epsilon_i)$, $\\epsilon_i\\sim N(0,0.03)$",
                   "near-identical (highly correlated) scores at any level"),
    "Duplicate": ("Int[5,12]", "$v\\sim U(0.1,0.4)$; $s_i=v$ for all $i$",
                  "exact repetition of one value"),
    "HeavyTailed": ("Int[2,12]", "$s_i=\\mathrm{clip}_{[0,1]}(0.15\\,X_i)$, $X_i\\sim$ Pareto-II(3)",
                    "mostly low scores, occasional large outlier"),
    "Uniform": ("Int[2,12]", "$s_i\\sim U(0,1)$ i.i.d.", "no structure"),
    "RandomBeta": ("Int[2,12]", "$a,b\\sim U(0.5,5)$; $s_i\\sim \\mathrm{Beta}(a,b)$ i.i.d.",
                   "random unimodal/U-shaped shapes"),
}

# ---------------------------------------------------------------------------
# Sub-threshold suite (all scores clipped to SIGNAL_CAP)
# ---------------------------------------------------------------------------
SIGNAL_CAP = 0.68


def _cap(a):
    return np.clip(a, 0.0, SIGNAL_CAP)


def h_tight_sub(rng):
    n = _n(rng, 3, 6)
    c = rng.uniform(0.10, 0.32)
    return _cap(c + rng.normal(0, 0.02, size=n))


def h_tight_near_review(rng):
    n = _n(rng, 3, 6)
    c = rng.uniform(0.22, 0.38)
    return _cap(c + rng.normal(0, 0.02, size=n))


def h_tight_near_remove(rng):
    n = _n(rng, 3, 6)
    c = rng.uniform(0.50, 0.66)
    return _cap(c + rng.normal(0, 0.015, size=n))


def h_loose_sub(rng):
    n = _n(rng, 3, 6)
    return _cap(rng.uniform(0.05, 0.35, size=n))


def h_loose_wide(rng):
    n = _n(rng)
    return _cap(rng.uniform(0.05, SIGNAL_CAP, size=n))


def h_diluted(rng):
    n_weak = _n(rng, 2, 6)
    weak = rng.uniform(0.02, 0.10, size=n_weak)
    border = rng.uniform(0.30, 0.40, size=1)
    return _cap(np.concatenate([weak, border]))


def h_many_weak(rng):
    n = _n(rng, 6, MAX_SIGNALS)
    return _cap(rng.uniform(0.05, 0.20, size=n))


def h_uniform_capped(rng):
    n = _n(rng)
    return _cap(rng.uniform(0.0, SIGNAL_CAP, size=n))


def h_duplicate_sub(rng):
    n = _n(rng, 4, MAX_SIGNALS)
    return _cap(np.full(n, rng.uniform(0.15, 0.32)))


SUB_SUITE = {
    "TightClusterSubReview": h_tight_sub,
    "TightClusterNearReview": h_tight_near_review,
    "TightClusterNearRemove": h_tight_near_remove,
    "LooseSpreadSubReview": h_loose_sub,
    "LooseSpreadWide": h_loose_wide,
    "DilutedBorderline": h_diluted,
    "ManyWeak": h_many_weak,
    "UniformCapped": h_uniform_capped,
    "DuplicateSubReview": h_duplicate_sub,
}

SUB_SPEC = {
    "TightClusterSubReview": ("Int[3,6]", "$c\\sim U(0.10,0.32)$; $s_i=c+\\epsilon_i$, $\\epsilon_i\\sim N(0,0.02)$"),
    "TightClusterNearReview": ("Int[3,6]", "$c\\sim U(0.22,0.38)$; $s_i=c+\\epsilon_i$, $\\epsilon_i\\sim N(0,0.02)$"),
    "TightClusterNearRemove": ("Int[3,6]", "$c\\sim U(0.50,0.66)$; $s_i=c+\\epsilon_i$, $\\epsilon_i\\sim N(0,0.015)$"),
    "LooseSpreadSubReview": ("Int[3,6]", "$s_i\\sim U(0.05,0.35)$ i.i.d."),
    "LooseSpreadWide": ("Int[2,12]", "$s_i\\sim U(0.05,0.68)$ i.i.d."),
    "DilutedBorderline": ("Int[2,6]+1", "weak $s_i\\sim U(0.02,0.10)$; one $s^*\\sim U(0.30,0.40)$"),
    "ManyWeak": ("Int[6,12]", "$s_i\\sim U(0.05,0.20)$ i.i.d."),
    "UniformCapped": ("Int[2,12]", "$s_i\\sim U(0,0.68)$ i.i.d."),
    "DuplicateSubReview": ("Int[4,12]", "$v\\sim U(0.15,0.32)$; $s_i=v$ for all $i$"),
}


def draw(suite: dict, n_trials: int, seed: int, reset_per_scenario: bool):
    """Draw score vectors for every scenario of `suite`.

    reset_per_scenario=False: one Generator(seed) shared by all scenarios, drawn
        in dict order (used for the main suite and the capture-rate run).
    reset_per_scenario=True: a fresh Generator(seed) per scenario (used for the
        parameter-sensitivity runs, so every parameter value sees identical draws).
    Returns {scenario: [np.ndarray, ...]}.
    """
    out = {}
    rng = np.random.default_rng(seed)
    for name, gen in suite.items():
        if reset_per_scenario:
            rng = np.random.default_rng(seed)
        out[name] = [gen(rng) for _ in range(n_trials)]
    return out
