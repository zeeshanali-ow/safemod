"""Supports the sentence after Theorem 3: maximum-signal monotonicity can fail once alpha > 1.

Raises the current maximum score of random inputs and counts cases where S_c decreases.
Usage: python tools/alpha_above_one_check.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from safemod.aggregation import proposed_operator, OperatorParams, Thresholds

def score(s, p, t):
    r = proposed_operator(np.asarray(s, float), p, t)
    return r[0] if isinstance(r, tuple) else r

def run(alpha, n=30000, seed=7):
    rng = np.random.default_rng(seed)
    viol, worst = 0, 0.0
    for _ in range(n):
        k = int(rng.integers(2, 10)); s = rng.uniform(0, 0.95, k)
        tr = rng.uniform(0.2, 0.6); tm = rng.uniform(tr + 0.05, 0.95)
        p = OperatorParams(f=rng.uniform(0, 1), K=int(rng.integers(1, 8)),
                           tau=rng.uniform(0.05, 0.3), alpha=alpha)
        t = Thresholds(tr, tm)
        i = int(np.argmax(s)); s2 = s.copy(); s2[i] = min(1.0, s[i] + rng.uniform(0, 0.1))
        d = score(s2, p, t) - score(s, p, t)
        if d < -1e-12:
            viol += 1; worst = min(worst, d)
    return viol, worst

if __name__ == "__main__":
    print("alpha  trials  decreases  largest_decrease")
    for a in (0.30, 1.0, 1.2, 1.5, 2.0):
        v, w = run(a)
        print(f"{a:5.2f}  30000  {v:9d}  {abs(w):.4f}")
