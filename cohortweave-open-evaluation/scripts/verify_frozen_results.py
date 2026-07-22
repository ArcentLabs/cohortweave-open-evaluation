#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
checks=[]
def load(name): return json.loads((root/'results'/name).read_text())
r=load('results.json'); i=load('results_imaging.json'); s=load('results_survival.json'); c=load('results_constraint_ablation.json'); ct=load('results_real_ct_pixel.json')
checks += [
 ('KS age', r['kpi1_fidelity']['ks_distance']['age_at_baseline'] <= 0.10),
 ('sex ratio after', abs(r['kpi2_bias_reduction']['imbalance_ratio_after']-1.0)<1e-12),
 ('PD-L1 MAE improvement', r['kpi3_clinical_completion']['pd_l1_MAE_improvement_pct'] >= 20),
 ('phantom Wasserstein reduction', i['kpi3_harmonization']['wasserstein_reduction_pct'] >= 30),
 ('phantom CCC', i['kpi3_harmonization']['joint_entropy_CCC_preserved'] >= 0.85),
 ('constraint violations repaired', c['headline']['constraint_proxy_contradiction_rate_pct'] == 0),
 ('fairness utility stress test', s['female_Cindex_gain'] > 0 and s['gap_reduction_pct'] > 0),
 ('real CT scope boundary', 'not official ICR' in ct['evidence_scope']),
 ('real CT patient-held-out split', 'leave-one-patient' in ct['split'].lower()),
]
for label,ok in checks: print(f"{'PASS' if ok else 'FAIL'}  {label}")
if not all(ok for _,ok in checks): raise SystemExit(1)
