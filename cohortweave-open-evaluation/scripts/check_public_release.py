#!/usr/bin/env python3
from pathlib import Path
from cohortweave_eval.release_audit import audit
root=Path(__file__).resolve().parents[1]
issues=audit(root)
if issues:
    print("PUBLIC RELEASE AUDIT FAILED")
    for issue in issues: print(f"- {issue}")
    raise SystemExit(1)
print("PUBLIC RELEASE AUDIT PASSED: no blocked medical files, model weights, application documents or obvious secrets found.")
