"""AI-BOOST Challenge 3 — PoC step 4: KPI metric-harness summary figure."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import wasserstein_distance

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
BASE = Path(__file__).resolve().parent
TEAL, CORAL, INK, GREY = "#1b7a7a", "#d1495b", "#22303a", "#9aa7ad"

real = pd.read_pickle(BASE / "cohort_real.pkl")
synth = pd.read_pickle(BASE / "cohort_synth.pkl")
aug = pd.read_pickle(BASE / "cohort_augmented.pkl")
R = json.load(open(BASE / "results.json"))
RI = json.load(open(BASE / "results_imaging.json"))

fig, ax = plt.subplots(2, 3, figsize=(13, 7.4))

# (1) Fidelity: age distribution real vs synthetic
a = ax[0, 0]
bins = np.linspace(29, 91, 30)
a.hist(real["age_at_baseline"], bins=bins, density=True, color=TEAL, alpha=0.55, label="Real")
a.hist(synth["age_at_baseline"], bins=bins, density=True, histtype="step",
       color=CORAL, lw=2, label="Synthetic")
a.set_title(f"KPI-1 Fidelity — Age\nKS = {R['kpi1_fidelity']['ks_distance']['age_at_baseline']} (target ≤ 0.10)")
a.set_xlabel("Age at baseline"); a.set_ylabel("Density"); a.legend(frameon=False)

# (2) Fidelity: categorical prevalence agreement (stage)
a = ax[0, 1]
order = ["I", "II", "III", "IV"]
pr = real["clinical_stage_group"].value_counts(normalize=True).reindex(order)
ps = synth["clinical_stage_group"].value_counts(normalize=True).reindex(order)
x = np.arange(len(order))
a.bar(x - 0.2, pr * 100, 0.4, color=TEAL, label="Real")
a.bar(x + 0.2, ps * 100, 0.4, color=CORAL, label="Synthetic")
a.set_xticks(x); a.set_xticklabels(order)
a.set_title("KPI-1 Fidelity — Clinical stage\n100% of categorical vars within ±5%")
a.set_xlabel("Stage group"); a.set_ylabel("Prevalence (%)"); a.legend(frameon=False)

# (3) Correlation preservation heatmap (abs error)
a = ax[0, 2]
cats = [c for c in real.columns if c not in
        ["subject_id", "date_baseline_ct", "clinical_staging_date", "date_smoking_status", "death_date", "age_at_baseline", "pd_l1"]]
def corr_num(df):
    d = df.copy()
    for c in cats:
        d[c] = d[c].astype("category").cat.codes
    keep = [c for c in df.columns if c not in ["subject_id", "date_baseline_ct", "clinical_staging_date", "date_smoking_status", "death_date"]]
    return d[keep].corr().to_numpy()
err = np.abs(corr_num(real) - corr_num(synth))
im = a.imshow(err, cmap="magma_r", vmin=0, vmax=0.3)
a.set_title(f"KPI-1 — |corr(real) − corr(synth)|\nmean abs error = {R['kpi1_fidelity']['mean_abs_correlation_error']}")
a.set_xticks([]); a.set_yticks([])
fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)

# (4) Bias reduction
a = ax[1, 0]
gb_before = real["gender"].value_counts().reindex(["male", "female"])
gb_after = aug["gender"].value_counts().reindex(["male", "female"])
x = np.arange(2)
a.bar(x - 0.2, gb_before, 0.4, color=GREY, label="Original")
a.bar(x + 0.2, gb_after, 0.4, color=TEAL, label="Augmented")
a.set_xticks(x); a.set_xticklabels(["Male", "Female"])
a.set_title(f"KPI-2 Bias — gender imbalance\n{R['kpi2_bias_reduction']['imbalance_ratio_before']}:1 → "
            f"{R['kpi2_bias_reduction']['imbalance_ratio_after']}:1")
a.set_ylabel("Subjects"); a.legend(frameon=False)

# (5) Clinical completion vs baseline
a = ax[1, 1]
comp = R["kpi3_clinical_completion"]["categorical"]
labels = {"clinical_stage_group": "Stage", "ecog_performance_status": "ECOG",
          "treatment_intent": "Tx intent", "metastasis_clinical_category": "M-cat",
          "distant_metastasis_pr": "Dist. met"}
keys = list(comp.keys())
x = np.arange(len(keys))
a.bar(x - 0.2, [comp[k]["mode_baseline"] * 100 for k in keys], 0.4, color=GREY, label="Baseline (mode)")
a.bar(x + 0.2, [comp[k]["model_accuracy"] * 100 for k in keys], 0.4, color=TEAL, label="Model")
a.axhline(90, ls="--", color=CORAL, lw=1, label="Target 90%")
a.set_xticks(x); a.set_xticklabels([labels[k] for k in keys], rotation=20, ha="right")
a.set_title("KPI-3 — Clinical completion accuracy")
a.set_ylabel("Accuracy (%)"); a.legend(frameon=False, fontsize=7.5)

# (6) Imaging: robustness + harmonization
a = ax[1, 2]
rob = RI["kpi4_robustness_CCC_joint_entropy"]
sig = [5, 10, 20]
ccc_vals = [rob[f"sigma_{s}HU"] for s in sig]
a.plot(sig, ccc_vals, "o-", color=TEAL, lw=2, label="GLCM entropy CCC")
a.axhline(0.85, ls="--", color=CORAL, lw=1, label="Target 0.85")
a.set_ylim(0.8, 1.005)
for s, v in zip(sig, ccc_vals):
    a.annotate(f"{v:.3f}", (s, v), textcoords="offset points", xytext=(0, 6), fontsize=7.5)
hr = RI["kpi3_harmonization"]
a.set_title(f"KPI-4 Robustness (CCC vs noise)\nKPI-3 harmoniz.: Wasserstein −{hr['wasserstein_reduction_pct']}%, "
            f"CCC {hr['joint_entropy_CCC_preserved']}")
a.set_xlabel("Gaussian noise σ (HU)"); a.set_ylabel("Concordance CCC")
a.set_xticks(sig); a.legend(frameon=False, fontsize=7.5)

fig.suptitle("AI-BOOST Challenge 3 — SPARK metric-harness PoC (surrogate cohort; CT phantoms; no imaging completion)",
             fontsize=11, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(BASE / "poc_results.png", bbox_inches="tight")
print("saved poc_results.png")
