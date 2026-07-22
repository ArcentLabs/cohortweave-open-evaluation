#!/usr/bin/env python3
"""Run the non-sensitive Spark PoC in an isolated work directory.

Default mode excludes the optional real-DICOM proxy and public-index QC.
Use --include-real-ct after installing requirements-real-ct.txt.
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments'/'spark'
CORE=['01_make_cohort.py','02_synth_eval.py','03_imaging.py','04_figures.py','05_survival_fairness.py','06_survival_figure.py','07_constraint_ablation.py','compute_budget.py']
OUTPUTS=['results.json','results_imaging.json','results_survival.json','results_constraint_ablation.json','compute_budget_output.txt','poc_results.png','poc_survival_fairness.png','poc_constraint_ablation.png']

def main():
 p=argparse.ArgumentParser(); p.add_argument('--include-real-ct',action='store_true'); p.add_argument('--output',type=Path,default=ROOT/'artifacts'/'latest'); a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='cohortweave-') as td:
  td=Path(td)
  for f in CORE+(['08_real_ct_pixel_evidence.py'] if a.include_real_ct else []): shutil.copy2(EXP/f,td/f)
  env=os.environ.copy(); env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
  for f in CORE+(['08_real_ct_pixel_evidence.py'] if a.include_real_ct else []):
   print(f'=== {f} ===',flush=True); subprocess.run([sys.executable,str(td/f)],check=True,cwd=td,env=env)
  outs=OUTPUTS+(['results_real_ct_pixel.json','results_real_ct_pixel_cases.csv'] if a.include_real_ct else [])
  for f in outs:
   if (td/f).exists(): shutil.copy2(td/f,a.output/f)
 print(f'Reproduction outputs written to {a.output}')
if __name__=='__main__': main()
