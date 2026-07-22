"""Run the portable Spark-phase PoC in sequence.

Set thread counts before importing scientific libraries to avoid excessive
parallelism on shared/local machines.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

SCRIPTS = [
    "01_make_cohort.py",
    "02_synth_eval.py",
    "03_imaging.py",
    "04_figures.py",
    "05_survival_fairness.py",
    "06_survival_figure.py",
    "07_constraint_ablation.py",
    "compute_budget.py",
]

if os.environ.get("COHORTWEAVE_INCLUDE_REAL_CT", "0") == "1":
    SCRIPTS.append("08_real_ct_pixel_evidence.py")

for script in SCRIPTS:
    print(f"\n=== {script} ===", flush=True)
    subprocess.run([sys.executable, str(BASE / script)], check=True, env=os.environ.copy())
