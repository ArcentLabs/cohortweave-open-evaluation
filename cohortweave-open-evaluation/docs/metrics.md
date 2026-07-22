# Evaluation metrics — definitions and implementations

This document separates **standard metrics** from **CohortWeave project-defined gates**. Core scalar implementations live in [`src/cohortweave_eval/metrics.py`](../src/cohortweave_eval/metrics.py). Image metrics in the public CT proxy use `scikit-image`; GLCM features use `skimage.feature`. The repository does **not** claim validation against PyRadiomics, and Lin's CCC is **not** treated as equivalent to ICC(3,1).

Targets labelled **ADVANCE target** are proposal acceptance criteria for protected-data validation. Spark values are engineering evidence only.

---

## 1. Two-sample Kolmogorov–Smirnov statistic

**Use:** compare continuous marginal distributions between reference and synthetic cohorts.

$$
D_{n,m}=\sup_x\left|F_n(x)-G_m(x)\right|
$$

where $F_n$ and $G_m$ are empirical cumulative distribution functions. Lower is better.

**Implementation:** `scipy.stats.ks_2samp`, wrapped by `ks_distance`.

**ADVANCE target:** $D_{n,m}\le 0.10$ for at least 80% of evaluated continuous variables.

**Limitation:** a low univariate KS statistic does not establish joint clinical fidelity or causal validity.

---

## 2. Mean absolute prevalence difference

**Use:** compare category frequencies for a categorical variable.

For category set $\mathcal{C}$:

$$
\Delta_{\mathrm{prev}}=
\frac{1}{|\mathcal{C}|}
\sum_{c\in\mathcal{C}}
\left|p_{\mathrm{synthetic}}(c)-p_{\mathrm{reference}}(c)\right|
$$

The public Spark scripts report the result in percentage points.

**Implementation:** `mean_prevalence_difference` and the frozen Spark experiment code.

**ADVANCE target:** prevalence difference $\le 5$ percentage points for at least 80% of evaluated categorical variables.

---

## 3. Wasserstein-1 distance and harmonisation reduction

**Use:** quantify one-dimensional distribution shift, including CT-intensity harmonisation proxies.

$$
W_1(P,Q)=\inf_{\gamma\in\Gamma(P,Q)}
\int |x-y|\,d\gamma(x,y)
$$

For the harmonisation gate:

$$
R_{W_1}=100\left(1-
\frac{W_1(P,Q_{\mathrm{harmonised}})}
{W_1(P,Q_{\mathrm{shifted}})}\right)
$$

**Implementation:** `scipy.stats.wasserstein_distance`, wrapped by `wasserstein_reduction`.

**ADVANCE target:** at least 30% reduction versus the unharmonised comparison.

**Limitation:** intensity-distribution alignment alone cannot establish anatomical or diagnostic preservation.

---

## 4. Lin's concordance correlation coefficient

**Use:** measure agreement, not only correlation, between paired radiomic-feature values.

$$
\rho_c=
\frac{2\,\operatorname{cov}(X,Y)}
{\operatorname{var}(X)+\operatorname{var}(Y)+(\mu_X-\mu_Y)^2}
$$

**Implementation:** `concordance_correlation_coefficient`, using population moments consistently with the Spark scripts.

**ADVANCE target:** CCC $\ge 0.85$ for selected pre-specified GLCM features.

**Important:** CCC and ICC(3,1) are related agreement statistics but are not generally interchangeable. The repository therefore does not claim ICC equivalence.

---

## 5. Structural similarity index

**Use:** local image-structure similarity in the public CT restoration proxy.

A canonical local form is:

$$
\operatorname{SSIM}(x,y)=
\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}
{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}
$$

**Implementation:** `skimage.metrics.structural_similarity` with `data_range=1.0` after per-image percentile normalisation to $[0,1]$.

**Spark proxy result:** TV denoising mean SSIM 0.833; 95% t-interval 0.808–0.859 across 36 corruption cases.

**Limitation:** SSIM is not evidence of lesion preservation or diagnostic equivalence.

---

## 6. Peak signal-to-noise ratio

**Use:** pixel-level reconstruction error relative to a declared image range.

$$
\operatorname{PSNR}=20\log_{10}\left(
\frac{L}{\sqrt{\operatorname{MSE}}}
\right)
$$

where $L$ is the evaluated data range.

**Implementation:** `skimage.metrics.peak_signal_noise_ratio` with $L=1.0$ for the current normalised public proxy. The repository does **not** hard-code 4095; that value would only be valid for an explicitly defined unsigned 12-bit stored-value analysis.

**Spark proxy result:** TV denoising mean PSNR 32.93 dB; 95% t-interval 32.28–33.57 dB across 36 corruption cases.

---

## 7. Sørensen–Dice overlap

**Use:** binary overlap for the public proxy's threshold-derived foreground mask.

$$
\operatorname{Dice}(A,B)=
\frac{2|A\cap B|}{|A|+|B|}
$$

**Implementation:** `dice_binary`; the CT proxy thresholds normalised pixels at 0.08.

**Spark proxy result:** mean foreground Dice 0.986; 95% t-interval 0.981–0.991.

**Important:** this is a **foreground structure-preservation proxy**, not organ or lesion segmentation validation.

---

## 8. Imbalance ratio and reduction

**Use:** quantify representation imbalance before and after steering.

For observed subgroup counts $n_g$:

$$
IR=\frac{\max_g n_g}{\min_g n_g}
$$

and

$$
IRR=100\left(1-\frac{IR_{\mathrm{post}}}{IR_{\mathrm{pre}}}\right).
$$

**Implementation:** `imbalance_ratio` and `imbalance_reduction_ratio`.

**Spark result:** sex ratio 2.135:1 $\rightarrow$ 1.0:1; ratio reduction 53.2%. The separate "excess imbalance removed" measure is 100% because parity removes all excess above 1.0.

**Limitation:** representation parity is not sufficient evidence of downstream fairness.

---

## 9. Constraint violation rate

**Use:** quantify records that violate at least one pre-specified clinical rule.

$$
CVR=100\times
\frac{N_{\mathrm{records\ with\ge 1\ violation}}}
{N_{\mathrm{records}}}
$$

**Implementation:** `constraint_violation_rate` and the rule matrix in the Spark constraint-ablation experiment.

**Spark results:** unconstrained generation 67.0% contradiction rate; constraint-repair proxy 0.0%. Post-hoc filtering produced 31.3% valid sample yield; yield is reported separately and is not itself a violation rate.

---

## 10. Non-sensitive fidelity preservation rate

**Use:** check whether fairness steering preserves distributions not directly targeted for balancing.

$$
FP=100\times
\frac{N_{\mathrm{non\text{-}sensitive\ variables\ within\ threshold}}}
{N_{\mathrm{non\text{-}sensitive\ variables}}}
$$

**Implementation:** `fidelity_preservation_rate`; the Spark gate uses a 5-percentage-point prevalence-drift threshold.

**Spark result:** 95.2% of evaluated non-sensitive categorical variables passed the threshold after fairness steering.

**Status:** project-defined evidence gate, not a universal fairness metric.

---

## 11. Relative MAE improvement

**Use:** compare model-based continuous-field completion with a baseline imputer.

$$
\Delta_{MAE}=100\left(
\frac{MAE_{\mathrm{baseline}}-MAE_{\mathrm{model}}}
{MAE_{\mathrm{baseline}}}
\right)
$$

**Implementation:** `relative_mae_improvement`.

**Spark result:** PD-L1 MAE improvement 32.2% versus median imputation on the deterministic surrogate.

**Limitation:** this is surrogate completion evidence, not protected clinical-data validation.

---

## 12. Perturbation robustness

**Use:** evaluate feature stability under pre-specified CT-like perturbations.

For feature vectors $f(I)$ and $f(\tilde I_{\delta})$:

$$
R_{\delta}=\rho_c\left(f(I),f(\tilde I_{\delta})\right)
$$

where $\delta$ identifies a declared perturbation level.

**Spark phantom result:** CCC 0.9998, 0.9979 and 0.9694 at 5, 10 and 20 HU-like perturbation levels.

**ADVANCE target:** CCC $\ge 0.85$ for the pre-specified feature set.

---

## 13. Harrell's concordance index and fairness utility gate

**Use:** evaluate ranking performance for right-censored survival outcomes overall and by subgroup.

$$
C=\frac{N_{\mathrm{concordant}}+0.5N_{\mathrm{tied\ risk}}}
{N_{\mathrm{comparable}}}
$$

**Implementation:** `c_index` and the Spark survival stress-test script.

**Spark surrogate result:** female C-index increased by 0.0715 and the absolute male–female C-index gap decreased by 36.3%.

**Release principle:** balancing is retained only if under-represented-group utility improves or remains non-inferior, overall utility is preserved, no other subgroup is materially harmed, and calibration is acceptable. Protected-data evaluation must use patient-level splits, sufficient subgroup size and bootstrap confidence intervals. The current synthetic survival stress test is not clinical validation.
