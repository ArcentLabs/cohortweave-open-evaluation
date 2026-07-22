# Utility-driven fairness stress test

Representation parity is not accepted as an end in itself. The Spark stress test deliberately creates severe female under-representation in a synthetic training set, trains a global Cox model and compares downstream survival ranking before and after synthetic augmentation.

Frozen surrogate result:

- female C-index: 0.4696 → 0.5410;
- male C-index: 0.6356 → 0.6468;
- overall C-index: 0.6026 → 0.6279;
- sex performance gap: 0.1660 → 0.1057 (36.3% reduction).

This result is a proof of evaluation design, not evidence that augmentation improves real patient outcomes.

## ADVANCE acceptance rule

Use patient-level held-out evaluation. Report subgroup C-index, calibration and bootstrap confidence intervals. Retain balancing only when the under-represented subgroup improves or is statistically non-inferior, overall utility is preserved and no other subgroup suffers material degradation. Otherwise reject or revise the balancing configuration.
