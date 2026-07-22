# Frozen Spark results

All values below are pre-access engineering evidence. Rows explicitly identify their data source and evidentiary limit.

| Evidence | Data source | Frozen result | Interpretation |
|---|---|---:|---|
| Continuous fidelity | Deterministic 1,088-row surrogate | KS ≈0.017–0.018 | Univariate surrogate smoke test |
| Representation | Surrogate | sex ratio 2.135:1 → 1.0:1 | Synthetic balancing demonstration |
| Continuous completion | Surrogate | PD-L1 MAE improvement 32.2% | Beats median imputation on the simulated target |
| Categorical completion | Surrogate | 2/5 targets ≥90% | Mixed result; official validation remains open |
| Constraint ablation | Surrogate | contradictions 67.0% → 0%; post-hoc valid yield 31.3% | Transparent proof of mechanism |
| Fairness utility stress test | Deliberately constructed synthetic survival outcome | female C-index +0.0715; gap −36.3% | Method signal, not clinical benefit |
| kVp-like harmonisation | CT-like phantoms | W1 reduction 94.0%; entropy CCC 0.9879 | Imaging metric harness validation |
| Robustness | CT-like phantoms | CCC 0.9998 / 0.9979 / 0.9694 at 5/10/20 HU | Phantom robustness only |
| Real-pixel restoration | Public pydicom test CT pixels; 5 patient IDs, 6 frames, 36 cases/method | TV PSNR 32.93 dB; SSIM 0.833; foreground Dice 0.986 | Small patient-held-out restoration proxy; not paired acquisition completion |
| ICR ≥95% | Protected multi-acquisition data required | Not measured | ADVANCE objective and principal technical risk |

## Fairness stress-test gate

The frozen surrogate result passes the illustrative gate because female, male and overall C-index improve and the sex gap narrows. On protected data, acceptance additionally requires patient-level splits, sufficient subgroup size, bootstrap confidence intervals and calibration. If balancing does not improve/preserve subgroup utility or harms another subgroup, the configuration is rejected.

## Negative result retained

A small residual CNN improved the corrupted public input but did not outperform TV denoising. This is retained to demonstrate baseline-first evaluation rather than assuming a deep model is superior.
