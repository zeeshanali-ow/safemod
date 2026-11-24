# SafeMod — Monte Carlo experiments (reproducibility package)

This package regenerates **every Monte Carlo table, figure and CSV** reported in the manuscript
*Policy-Governed Evidence Fusion for Deterministic Content Moderation* from one command.
Everything is deterministic: same code + same seed ⇒ identical numbers on any machine.

## Quick start

### Option A — local (VS Code or any terminal)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python tests/test_operator.py          # 9 tests, ~15 s
python run_all.py --quick              # smoke test, ~15 s
python run_all.py                      # full run, about 10-20 min on a laptop
```

### Option B — Google Colab
Upload this package as a ZIP, then run the cells in `SafeMod_MonteCarlo_Colab.ipynb`
(or paste, adjusting the folder name):
```python
!unzip -o -q R3_code_and_results.zip -d /content/r3
%cd /content/r3/safemod
!pip install -q -r requirements.txt
!python tests/test_operator.py
!python run_all.py
!zip -r -q outputs.zip outputs     # then download outputs.zip from the Files panel
```

Recommendation: run locally. It is a few CPU minutes, needs no GPU, and keeps the
environment fixed. Colab is fine too; record its Python/numpy versions (they are written
to `outputs/manifest.json` automatically).

## What is run

| ID | Experiment | Size | Paper element |
|----|------------|------|---------------|
| E1 | Main suite: 8 scenarios, MAX / Mean / Noisy-OR / proposed | 8 × 10,000 | Tables IV, VII; Figs. 2–3 |
| E2 | Sub-threshold suite: 9 scenarios, every score < θm | 9 × 10,000 | Tables V, VIII; Fig. 4 |
| E3 | Sensitivity: min/default/max table, finite differences, 9-point curves, K–α and τ–α grids | 90,000 per point | Tables IX–X; Fig. 5 |
| E4 | All 15 presets (range midpoints) on the main suite, with each preset's escalation zone | 15 × 80,000 | Suppl. Table S1 |
| E5 | Coordinate-wise perturbation: one score raised by δ ~ U(0, 0.10) | 15 × 30,000 = 450,000 | Section III-C; Suppl. Table S3 |

Fixed settings (all in `safemod/config.py`): seed 42 (perturbation seed 2026), θr = 0.30,
θm = 0.70, reference preset 3b (Zero-Tolerance Illegal) at its range midpoint:
f = 0.20, K = 5, τ = 0.10, α = 0.20.

## Outputs (`outputs/`)

- `csv/` — every result as CSV; `*_trials.csv.gz` hold one row per trial (all scores and decisions).
- `latex/` — booktabs tables to `\input{}` directly: `tab_main_spec`, `tab_main_overall`,
  `tab_main_vs_max`, `tab_sub_spec`, `tab_sub_vs_max`, `tab_sensitivity`, `tab_finite_differences`,
  `tab_presets`, `tab_perturbation`.
- `figures/` — PNG (300 dpi) and PDF of every figure.
- `checks.csv` — structural guarantees checked on **every trial** (all columns must be 0):
  S_c below MAX, S_c above the uplift bound, Remove decision differing from MAX, Review→Allow vs MAX.
- `reproduction.csv` — full run only: compares the newly computed values with those already
  published in the previous manuscript version (sub-threshold escalation rates and decision-sensitivity table). Expected: all exact matches.
- `manifest.json` — Python/numpy/pandas versions, platform, settings, timings and the SHA-256 of
  every output file. Commit it with the results.

## Code map

| File | Content |
|------|---------|
| `safemod/aggregation.py` | MAX, mean, Noisy-OR and the proposed operator (Eqs. of Section III, Algorithm 1); decision rule; uplift bound; escalation zone |
| `safemod/scenarios.py` | all 17 generators with their specification strings (the spec tables are generated from these) |
| `safemod/config.py` | seeds, N, thresholds, the 15 presets (Table II ranges), sensitivity points and grids |
| `safemod/experiments.py` | E1–E5 and the structural checks |
| `safemod/reporting.py` | LaTeX tables and figures |
| `run_all.py` | single entry point |
| `tests/test_operator.py` | worked examples of Table III, single/duplicate examples, noise-floor counterexample, uplift bound, Theorem 3, threshold boundaries |

## Implementation notes

- **Numerics.** All computation, decisions and checks use full double precision; nothing is
  rounded inside the operator. S_base is evaluated as u_max − Σ wᵢ(u_max − uᵢ), algebraically
  identical to Σ wᵢuᵢ, so S_base ≤ u_max holds exactly in floating point. Checks are exact except
  the uplift bound (1e-12). An earlier version rounded S_c to 6 decimals; see
  `results_history/NUMERICAL_FIX.md` for the correction and its (negligible) effect.
- **Random-number streams.** E1, E2 and E4 draw scenarios from one generator in table order;
  E3 resets the generator for each scenario so every parameter value sees identical score vectors.
  This is why the default Allow share in E3 (34.05%) differs slightly from E2 (34.23%).
- **Allow→Review = extra review volume.** Because S_c ≥ max_i s_i and Remove decisions coincide
  with MAX, the only possible decision change versus MAX is Allow→Review, so its count is exactly
  the additional human-review load.
- **Uplift bound.** Every trial satisfies max_i s_i ≤ S_c ≤ max_i s_i + α(θm − max_i s_i)⁺,
  and Allow→Review can occur only when (θr − αθm)/(1 − α) ≤ max_i s_i < θr.

---

# Jigsaw real-data evaluation (`jigsaw/`)

Two steps, so that the expensive model run happens once and everything after it is
deterministic and checkable from a released predictions file.

| Step | Script | Where | Time |
|------|--------|-------|------|
| 1. Score | `jigsaw/score_jigsaw.py` | Colab GPU (T4) | ~5-15 min |
| 2. Evaluate | `jigsaw/eval_jigsaw.py` | anywhere (CPU) | ~2-5 min |

Open `SafeMod_Jigsaw_Colab.ipynb` in Colab and run it top to bottom; it handles the data download
(Kaggle API or manual), scoring, evaluation and the final download.

**Data.** Jigsaw Toxic Comment Classification Challenge, `test.csv` + `test_labels.csv`; rows labelled
−1 (not used for scoring in the challenge) are removed, leaving 63,978 comments. You must accept the
competition rules on Kaggle once before downloading.

**Model.** Detoxify `original` (bert-base-uncased fine-tuned on the challenge's *training* split), so the
test split is unseen by the model. Version 0.5.4, checkpoint `toxic_original-c1212f89`; its file name and SHA-256 are recorded in
`scoring_manifest.json`.

**Evaluation settings** (top of `eval_jigsaw.py`):
- Category mappings: *Toxicity* = all six labels (ground truth: any label is 1); *Harassment* =
  {toxic, insult, identity_hate, threat} (ground truth: insult, identity_hate or threat). Each label
  appears at most once per category.
- Threshold policies (θr/θm): Highly Cautious 0.20/0.50, Cautious 0.35/0.70, Default 0.40/0.75,
  Balanced 0.50/0.80, Permissive 0.55/0.85.
- Presets: 11 (General-Purpose Default) and 3b (Zero-Tolerance Illegal) at their range midpoints;
  all 15 presets at the Default policy for the supplementary table.

**Metrics** (all over the same N = 63,978 items): decision shares; Allow→Review count (= extra review
volume) and its precision (share of escalated items that are labelled violations); flagged
(Review or Remove) precision, recall, FPR, FNR; Remove precision/recall; ROC AUC, Brier score and ECE of
each aggregated score; 95% Wilson intervals throughout; per-label quality of the scoring model.
Structural checks run on every item (`checks.csv`, all zeros expected).

**Outputs** (`jigsaw_outputs/`): `jigsaw_predictions.csv` (ids, scores, labels; no comment text),
`scoring_manifest.json`, and `eval/` with `csv/`, `latex/`, `figures/`, `checks.csv`, `manifest.json`.
The dataset is released under CC0; the predictions file contains no comment text (see `outputs/jigsaw_scoring/DATA_NOTICE.md`).

Pipeline test without the model or data: `python jigsaw/score_jigsaw.py --mock --limit 2000`
then `python jigsaw/eval_jigsaw.py --pred jigsaw_outputs/jigsaw_predictions.csv`.

## Supporting check: alpha > 1
`python tools/alpha_above_one_check.py` raises the maximum score of 30,000 random inputs per alpha value and counts decreases of S_c. No decrease occurs for alpha = 0.30 or 1.0; decreases appear from alpha = 1.2 (output in `results_history/alpha_above_one_check_output.txt`). This supports the remark after Theorem 3 that alpha <= 1 is required.
