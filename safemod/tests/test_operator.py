"""
Unit tests: the operator against the manuscript's worked examples and its
proven properties.   Run:  python -m pytest -q   (or)   python tests/test_operator.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safemod.aggregation import (OperatorParams, Thresholds, proposed_operator, decide,
                                 max_pool, mean_pool, noisy_or, uplift_bound, escalation_zone, BOUND_TOL)

# Worked-example configuration of Section III-F (Table II)
P = OperatorParams(f=0.7, K=3, tau=0.12, alpha=0.10)
T = Thresholds(theta_r=0.65, theta_m=0.85)
CASES = {
    1: [0.2] * 7,
    2: [0.2, 0.3, 0.4, 0.68, 0.685, 0.69, 0.695],
    3: [0.3, 0.4, 0.5, 0.78, 0.81, 0.87, 0.9],
    4: [0.2, 0.3, 0.4, 0.6, 0.645, 0.647, 0.649],
}


def close(a, b, tol=5e-5):
    return abs(a - b) <= tol


def test_table2_proposed():
    assert close(proposed_operator(CASES[1], P, T), 0.20)
    assert close(proposed_operator(CASES[2], P, T), 0.7061)
    assert close(proposed_operator(CASES[3], P, T), 0.90)
    assert close(proposed_operator(CASES[4], P, T), 0.6572)
    assert decide(proposed_operator(CASES[4], P, T), T) == "Review"    # escalated vs MAX (Allow)


def test_table2_baselines():
    assert close(mean_pool(CASES[2]), 0.5214) and close(mean_pool(CASES[3]), 0.6514)
    assert close(mean_pool(CASES[4]), 0.4916)
    assert close(noisy_or(CASES[1]), 0.7903) and close(noisy_or(CASES[2]), 0.9968)
    assert max_pool(CASES[4]) == 0.649


def test_single_and_duplicate_examples():
    assert close(proposed_operator([0.649], P, T), 0.65235)
    one = decide(proposed_operator([0.645], P, T), T)
    two = decide(proposed_operator([0.645, 0.645], P, T), T)
    assert one != two          # duplicate sensitivity example of the manuscript


def test_empty_and_fallback():
    assert proposed_operator([], P, T) == 0.0
    assert proposed_operator([0.1, 0.2], P, T) == 0.2          # all below rho = 0.455 -> MAX fallback


def test_r1_counterexample_non_monotone():
    p = OperatorParams(f=0.2, K=5, tau=0.10, alpha=0.25)
    t = Thresholds(theta_r=0.65, theta_m=0.85)
    a = proposed_operator([0.20, 0.30, 0.70], p, t)
    b = proposed_operator([0.25, 0.30, 0.70], p, t)
    assert close(a, 0.70561) and close(b, 0.70422) and b < a   # conceded in the manuscript


def test_bound_and_max_floor_random():
    rng = np.random.default_rng(1)
    for _ in range(20_000):
        tr = rng.uniform(0.05, 0.9)
        t = Thresholds(theta_r=tr, theta_m=rng.uniform(tr + 0.01, 1.0))
        p = OperatorParams(f=rng.uniform(0, 1), K=int(rng.integers(1, 9)),
                           tau=rng.uniform(0.01, 1.0), alpha=rng.uniform(0, 0.30))
        s = rng.uniform(0, 1, int(rng.integers(1, 15)))
        S, M = proposed_operator(s, p, t), float(s.max())
        assert S >= M                                          # Property 3 (raw-maximum floor), exact
        assert S <= uplift_bound(M, p, t) + BOUND_TOL          # Proposition (worst-case uplift)
        assert (S >= t.theta_m) == (M >= t.theta_m)            # Remove iff some s_i >= theta_m
        lo, hi = escalation_zone(p, t)
        if decide(M, t) == "Allow" and decide(S, t) == "Review":
            assert lo - BOUND_TOL <= M < hi                    # escalation only inside the zone


def test_theorem4_raising_the_max():
    rng = np.random.default_rng(7)
    t = Thresholds(theta_r=0.30, theta_m=0.70)
    for _ in range(20_000):
        p = OperatorParams(f=rng.uniform(0, 1), K=int(rng.integers(1, 9)),
                           tau=rng.uniform(0.05, 0.3), alpha=rng.uniform(0, 0.30))
        s = rng.uniform(0, 1, int(rng.integers(1, 12)))
        j = int(np.argmax(s))
        s2 = s.copy()
        s2[j] = min(1.0, s[j] + rng.uniform(0, 0.1))
        assert proposed_operator(s2, p, t) >= proposed_operator(s, p, t)   # exact


def test_threshold_boundaries_exact():
    """Scores one floating-point step below a threshold must never be pushed across
    theta_m, and S_c must never fall below max_i s_i - checked without tolerance."""
    for tm in (0.5, 0.7, 0.75, 0.85, 1.0):
        for tr in (0.2, 0.3, 0.4):
            t = Thresholds(theta_r=tr, theta_m=tm)
            below_m = np.nextafter(tm, 0.0)
            for alpha in (0.0, 0.1, 0.2, 0.3):
                for K in (1, 3, 5, 8):
                    for f in (0.0, 0.2, 0.7):
                        p = OperatorParams(f=f, K=K, tau=0.10, alpha=alpha)
                        for n in (1, 2, K, K + 3):
                            s = [below_m] * n
                            S = proposed_operator(s, p, t)
                            assert S >= below_m and S < tm and decide(S, t) == "Review"
                            s = [below_m] * n + [tm]
                            assert decide(proposed_operator(s, p, t), t) == "Remove"
                            s = [np.nextafter(tr, 0.0)] * n
                            S = proposed_operator(s, p, t)
                            assert S >= s[0] and S < tm


def test_remove_equivalence_exact_random():
    rng = np.random.default_rng(11)
    for _ in range(50_000):
        tr = rng.uniform(0.05, 0.9)
        t = Thresholds(theta_r=tr, theta_m=rng.uniform(tr + 0.01, 1.0))
        p = OperatorParams(f=rng.uniform(0, 1), K=int(rng.integers(1, 9)),
                           tau=rng.uniform(0.01, 1.0), alpha=rng.uniform(0, 0.30))
        n = int(rng.integers(1, 12))
        # concentrate scores near theta_m to stress the boundary
        s = np.clip(t.theta_m + rng.normal(0, 1e-9 if rng.random() < 0.5 else 0.02, n), 0, 1)
        S, M = proposed_operator(s, p, t), float(s.max())
        assert S >= M
        assert (decide(S, t) == "Remove") == (M >= t.theta_m)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("all tests passed")
