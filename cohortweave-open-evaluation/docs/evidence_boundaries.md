# Spark versus ADVANCE evidence boundary

## Demonstrated in Spark

- deterministic full-schema surrogate generation;
- tabular fidelity, completion, balancing and privacy smoke-test harnesses;
- constraint ablation and evidence-gate logic;
- synthetic CT phantom harmonisation/robustness metrics;
- small patient-held-out public real-pixel restoration proxy;
- illustrative downstream fairness utility stress test;
- portable scripts, tests and machine-readable results.

## Not demonstrated in Spark

- performance on protected EUCAIM clinical data or complete CT studies;
- paired same-patient non-contrast/contrast/low-dose acquisition generation;
- official ICR ≥95%;
- radiologist-established plausibility or diagnostic equivalence;
- privacy safety sufficient for unrestricted release;
- clinical deployment or regulatory conformity.

## ADVANCE critical path

Secure inventory determines whether the complete multi-acquisition subset is adequate. Deterministic and 2.5D baselines precede diffusion. A month-2 go/no-go requires a complete held-out benchmark and measurable improvement over deterministic baselines. Full 3D completion is a stretch goal. If the gate fails, effort moves to tabular synthesis/completion/balancing and kVp harmonisation while completion failures and abstentions are reported transparently.
