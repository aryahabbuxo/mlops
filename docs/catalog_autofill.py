"""Fill measurable catalog fields from local files.

Usage:  python catalog_autofill.py dataset_catalog.csv [project_root]

For each row whose local_path exists under project_root it fills
n_files, n_rows, n_cols (first sheet / csv, raw read) and sha256_all_files
(hash over the sorted per-file hashes). Existing non-TO_FILL values for
n_rows / n_cols are left alone so hand-verified values are never overwritten.
Rows whose path is missing are left untouched and reported.
"""
import csv
import hashlib
import os
import sys

import pandas as pd

TO_FILL = "TO_FILL"
DATA_EXT = {".csv", ".xlsx", ".xls"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_files(path):
    if os.path.isfile(path):
        return [path]
    out = []
    for base, _, names in os.walk(path):
        out += [os.path.join(base, n) for n in names]
    return sorted(out)


def shape_of(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".csv":
            with open(path, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            i = next((k for k, line in enumerate(lines) if line.startswith("YEAR")), 0)
            df = pd.read_csv(path, skiprows=i)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            return None
    except Exception as exc:
        print(f"[warn] could not read {path}: {exc}")
        return None
    return df.shape


def main(catalog, root="."):
    with open(catalog, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
        cols = list(rows[0].keys())
    for r in rows:
        p = os.path.join(root, r["local_path"])
        if r["local_path"] == TO_FILL or not os.path.exists(p):
            print(f"[skip] {r['id']}: path not found ({r['local_path']})")
            continue
        files = list_files(p)
        data_files = [f for f in files if os.path.splitext(f)[1].lower() in DATA_EXT]
        r["n_files"] = str(len(files))
        r["sha256_all_files"] = hashlib.sha256(
            "".join(sha256_file(f) for f in files).encode()).hexdigest()
        shapes = [s for s in (shape_of(f) for f in data_files) if s]
        if shapes:
            if r["n_rows"] in ("", TO_FILL):
                r["n_rows"] = str(sum(s[0] for s in shapes))
            if r["n_cols"] in ("", TO_FILL):
                r["n_cols"] = str(max(s[1] for s in shapes))
        print(f"[ok]   {r['id']}: {len(files)} files, rows={r['n_rows']}, cols={r['n_cols']}")
    with open(catalog, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ".")
