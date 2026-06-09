"""
Strip all outputs and execution counts from Jupyter notebooks.

Use this to reduce file size and memory pressure in PyCharm's notebook
viewer — embedded PNG outputs in a fully-run analysis notebook can add
up to several MB of base64 blobs that PyCharm decodes into RAM.

Usage (from project root):
    python scripts/strip_notebook_outputs.py notebooks/02b_analysis_topo.ipynb
    python scripts/strip_notebook_outputs.py notebooks/*.ipynb

Also prints the size reduction per file.
"""

import json
import os
import sys


def strip(path: str) -> None:
    size_before = os.path.getsize(path)
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cleared_outputs = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                cleared_outputs += len(cell["outputs"])
            cell["outputs"] = []
            cell["execution_count"] = None

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    size_after = os.path.getsize(path)
    mb_before = size_before / 1024 / 1024
    mb_after = size_after / 1024 / 1024
    print(f"{path}: {mb_before:.2f} MB -> {mb_after:.2f} MB "
          f"({cleared_outputs} outputs cleared)")


def main(argv):
    if len(argv) < 2:
        print("usage: strip_notebook_outputs.py <notebook.ipynb> [more.ipynb ...]")
        sys.exit(1)
    for p in argv[1:]:
        if not p.endswith(".ipynb"):
            print(f"skip (not .ipynb): {p}")
            continue
        if not os.path.isfile(p):
            print(f"skip (not found): {p}")
            continue
        strip(p)


if __name__ == "__main__":
    main(sys.argv)
