"""
compare_outputs.py - compare two result folders (e.g. before/after a code change).

    python tools/compare_outputs.py OLD_DIR NEW_DIR [--report report.md]

For every CSV present in both folders it reports whether integer columns
(counts, decisions) are identical and the largest absolute difference in
floating-point columns. Trial-level files (*.csv.gz) are compared row by row.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compare(old: Path, new: Path):
    rows = []
    for f in sorted(old.rglob("*.csv*")):
        rel = f.relative_to(old)
        g = new / rel
        if not g.exists():
            rows.append({"file": str(rel), "status": "missing in new"}); continue
        a, b = pd.read_csv(f), pd.read_csv(g)
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            rows.append({"file": str(rel), "status": f"shape/columns differ {a.shape} vs {b.shape}"}); continue
        ints = [c for c in a.columns if pd.api.types.is_integer_dtype(a[c]) and pd.api.types.is_integer_dtype(b[c])]
        flts = [c for c in a.columns if c not in ints and pd.api.types.is_numeric_dtype(a[c])]
        txt = [c for c in a.columns if c not in ints and c not in flts]
        int_diff = int((a[ints] != b[ints]).to_numpy().sum()) if ints else 0
        txt_diff = int((a[txt].astype(str) != b[txt].astype(str)).to_numpy().sum()) if txt else 0
        fd = float(np.nanmax(np.abs(a[flts].to_numpy(float) - b[flts].to_numpy(float)))) if flts else 0.0
        worst = ""
        if flts and fd > 0:
            d = np.abs(a[flts].to_numpy(float) - b[flts].to_numpy(float))
            i, j = np.unravel_index(np.nanargmax(d), d.shape)
            worst = f"{flts[j]} row {i}: {a[flts[j]].iat[i]!r} -> {b[flts[j]].iat[i]!r}"
        rows.append({"file": str(rel), "rows": len(a), "int_cells_changed": int_diff,
                     "text_cells_changed": txt_diff, "max_float_diff": fd, "largest_change": worst,
                     "status": "identical" if int_diff == txt_diff == 0 and fd == 0 else "changed"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("old"); ap.add_argument("new"); ap.add_argument("--report", default=None)
    a = ap.parse_args()
    df = compare(Path(a.old), Path(a.new))
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 80)
    print(df.to_string(index=False))
    if a.report:
        Path(a.report).write_text(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False))
