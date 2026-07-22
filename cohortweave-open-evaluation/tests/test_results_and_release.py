import json
from pathlib import Path
from cohortweave_eval.release_audit import audit
ROOT=Path(__file__).resolve().parents[1]
def test_evidence_boundary():
    d=json.loads((ROOT/'results/results_real_ct_pixel.json').read_text())
    assert 'not official ICR' in d['evidence_scope']
    assert d['unique_patient_ids']==5 and d['evaluations_per_method']==36
def test_public_release_audit(): assert audit(ROOT)==[]
