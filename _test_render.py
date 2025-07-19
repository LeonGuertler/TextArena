#!/usr/bin/env python3
"""
chargrid_numbers_only.py
Convert an ARC‑AGI‑3 frame to a 1‑char‑per‑cell ASCII grid
with *only* the 10→A, 11→B, … replacement.

Run:
    python chargrid_numbers_only.py           # uses tmp.json
    python chargrid_numbers_only.py my.json   # custom file
"""
import json, sys
from pathlib import Path
from typing import List

# 1️⃣  load ----------------------------------------------------------------
def load_frame(path: str) -> List[List[int]]:
    with open(path) as f:
        data = json.load(f)
    return data[0] if isinstance(data[0][0], list) else data

frame_file = sys.argv[1] if len(sys.argv) > 1 else "tmp.json"
frame = load_frame(frame_file)

# 2️⃣  single‑rule glyph mapper -------------------------------------------
def glyph(v: int) -> str:
    if v < 10:
        return str(v)           # keep 0‑9 as is
    return chr(ord("A") + v - 10)  # 10→A, 11→B, …

# 3️⃣  render --------------------------------------------------------------
ascii_grid = "\n".join("".join(glyph(c) for c in row) for row in frame)
print(ascii_grid)

Path("frame_chargrid_numbers_only.txt").write_text(ascii_grid)
print("\nSaved to frame_chargrid_numbers_only.txt")
