"""
score_jigsaw.py  -  Step 1 (GPU recommended): score the Jigsaw test set once.

Scores every labelled comment of the Jigsaw Toxic Comment Classification
Challenge test set (test.csv + test_labels.csv, rows labelled -1 removed:
63,978 comments) with the public Detoxify 'original' model (bert-base-uncased
fine-tuned on the challenge's TRAINING split, so the test split is unseen).

Output (in --out, default jigsaw_outputs/):
  jigsaw_predictions.csv   id, the 6 model scores, the 6 ground-truth labels
                           (no comment text is stored)
  scoring_manifest.json    model, checkpoint SHA-256, library versions, device,
                           input/output SHA-256, row counts, timing

Usage (Colab, GPU runtime):
  python jigsaw/score_jigsaw.py --data data/ --out jigsaw_outputs
  python jigsaw/score_jigsaw.py --mock --limit 500     # pipeline test without the model

The script saves progress every --checkpoint-every rows and resumes if
interrupted (Colab disconnects): just run the same command again.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
# Detoxify 'original' output name -> Jigsaw label name
DETOXIFY_TO_JIGSAW = {"toxicity": "toxic", "toxic": "toxic",
                      "severe_toxicity": "severe_toxic", "severe_toxic": "severe_toxic",
                      "obscene": "obscene", "threat": "threat", "insult": "insult",
                      "identity_attack": "identity_hate", "identity_hate": "identity_hate"}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_test(data_dir: Path) -> pd.DataFrame:
    test = pd.read_csv(data_dir / "test.csv")
    lab = pd.read_csv(data_dir / "test_labels.csv")
    df = test.merge(lab, on="id", how="inner", validate="one_to_one")
    keep = (df[LABELS] != -1).all(axis=1)
    df = df.loc[keep, ["id", "comment_text"] + LABELS].reset_index(drop=True)
    return df


class MockModel:
    """Deterministic stand-in used only by --mock to test the pipeline."""
    class_names = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]

    def predict(self, texts):
        rng = np.random.default_rng(len("".join(texts)) % 2**32)
        return {k: rng.beta(0.3, 3.0, len(texts)).tolist() for k in self.class_names}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="folder with test.csv and test_labels.csv")
    ap.add_argument("--out", default="jigsaw_outputs")
    ap.add_argument("--model", default="original", help="Detoxify model name (default: original)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--checkpoint-every", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N rows (testing)")
    ap.add_argument("--mock", action="store_true", help="use a random stand-in model (pipeline test)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "jigsaw_predictions.partial.csv"
    final = out / "jigsaw_predictions.csv"
    t0 = time.time()

    if args.mock:
        rng = np.random.default_rng(0)
        n = args.limit or 2000
        df = pd.DataFrame({"id": [f"mock{i:06d}" for i in range(n)],
                           "comment_text": ["x" * int(k) for k in rng.integers(5, 200, n)]})
        for l in LABELS:
            df[l] = (rng.random(n) < 0.06).astype(int)
        model, device, versions = MockModel(), "cpu", {"detoxify": "MOCK"}
        ckpt = {"file": "MOCK", "sha256": "MOCK"}
        inputs = {}
    else:
        data = Path(args.data)
        df = load_test(data)
        if args.limit:
            df = df.iloc[: args.limit].copy()
        import torch
        from detoxify import Detoxify
        import detoxify as _dx
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("WARNING: no GPU found - scoring will be slow (hours).", flush=True)
        model = Detoxify(args.model, device=device)
        hub = Path(torch.hub.get_dir()) / "checkpoints"
        cks = sorted(glob.glob(str(hub / f"*{args.model}*.ckpt")))
        ckpt = {"file": os.path.basename(cks[0]) if cks else "not found",
                "sha256": sha256(cks[0]) if cks else "not found"}
        import transformers
        from importlib.metadata import version as _pkgver
        versions = {"detoxify": _pkgver("detoxify"), "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "none"}
        inputs = {"test.csv": sha256(data / "test.csv"), "test_labels.csv": sha256(data / "test_labels.csv")}

    print(f"Rows to score: {len(df):,}  (device: {device})", flush=True)

    done = pd.read_csv(partial) if partial.exists() else None
    done_ids = set(done["id"]) if done is not None else set()
    todo = df[~df["id"].isin(done_ids)]
    if done_ids:
        print(f"Resuming: {len(done_ids):,} already scored, {len(todo):,} left", flush=True)

    # Sort by length so batches pad little; results are re-ordered by id at the end.
    todo = todo.assign(_len=todo["comment_text"].str.len()).sort_values("_len", kind="stable")
    buf, scored_since = [], 0
    for start in range(0, len(todo), args.batch_size):
        chunk = todo.iloc[start: start + args.batch_size]
        res = model.predict(chunk["comment_text"].tolist())
        rec = pd.DataFrame({"id": chunk["id"].to_numpy()})
        for k, v in res.items():
            if k in DETOXIFY_TO_JIGSAW:
                rec[f"p_{DETOXIFY_TO_JIGSAW[k]}"] = np.asarray(v, dtype=float)
        for l in LABELS:
            rec[l] = chunk[l].to_numpy()
        buf.append(rec)
        scored_since += len(chunk)
        if scored_since >= args.checkpoint_every or start + args.batch_size >= len(todo):
            new = pd.concat(buf, ignore_index=True)
            new.to_csv(partial, mode="a", header=not partial.exists(), index=False, float_format="%.10g")
            buf, scored_since = [], 0
            n_done = len(done_ids) + start + len(chunk)
            print(f"  {n_done:,}/{len(df):,} scored  ({time.time() - t0:,.0f} s)", flush=True)

    pred = pd.read_csv(partial)
    missing = [f"p_{l}" for l in LABELS if f"p_{l}" not in pred.columns]
    if missing:
        sys.exit(f"Model did not return scores for: {missing}. Check the model name.")
    order = {i: k for k, i in enumerate(df["id"])}
    pred = pred.drop_duplicates("id").sort_values("id", key=lambda s: s.map(order)).reset_index(drop=True)
    assert len(pred) == len(df), f"expected {len(df)} rows, got {len(pred)}"
    cols = ["id"] + [f"p_{l}" for l in LABELS] + LABELS
    pred[cols].to_csv(final, index=False, float_format="%.10g")
    partial.unlink()

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"), "mock": args.mock,
        "model": f"Detoxify '{args.model}'", "checkpoint": ckpt, "versions": versions,
        "python": sys.version.split()[0], "platform": platform.platform(), "device": device,
        "batch_size": args.batch_size, "rows": len(pred),
        "inference": "Detoxify.predict on raw comment_text; Detoxify tokenizer defaults (truncation=True, "
                     "padding=True); model in eval mode; no other preprocessing; rows with any -1 label removed",
        "command": " ".join(sys.argv),
        "positives_any_label": int((pred[LABELS].max(axis=1) == 1).sum()),
        "label_positives": {l: int(pred[l].sum()) for l in LABELS},
        "inputs_sha256": inputs, "output_sha256": sha256(final),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "scoring_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done: {final} ({len(pred):,} rows). Manifest written.", flush=True)


if __name__ == "__main__":
    main()
