"""AI-BOOST Challenge 3 — survival + fairness figure."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
BASE = Path(__file__).resolve().parent
TEAL, CORAL, INK, GREY = "#1b7a7a", "#d1495b", "#22303a", "#9aa7ad"
R = json.load(open(BASE / "results_survival.json"))
b, e = R["baseline_original_cohort"], R["enhanced_augmented_cohort"]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

# panel 1: C-index by sex, baseline vs enhanced
a = ax[0]
groups = ["male", "female", "overall"]
x = np.arange(len(groups))
a.bar(x - 0.2, [b[g] for g in groups], 0.4, color=GREY, label="Original (10:1)")
a.bar(x + 0.2, [e[g] for g in groups], 0.4, color=TEAL, label="Enhanced (1:1)")
a.axhline(0.5, ls=":", color=INK, lw=1)
a.text(2.35, 0.505, "random", fontsize=8, color=INK)
for i, g in enumerate(groups):
    a.annotate(f"{b[g]:.2f}", (i - 0.2, b[g]), ha="center", va="bottom", fontsize=8, color=INK)
    a.annotate(f"{e[g]:.2f}", (i + 0.2, e[g]), ha="center", va="bottom", fontsize=8, color=INK)
a.set_xticks(x); a.set_xticklabels(["Men", "Women", "Overall"])
a.set_ylim(0, 0.85); a.set_ylabel("Survival C-index")
a.set_title("Prognostic accuracy by sex\n(women = under-represented subgroup)")
a.legend(frameon=False, fontsize=9)

# panel 2: sex gap reduction
a = ax[1]
a.bar([0, 1], [b["gap"], e["gap"]], 0.5, color=[CORAL, TEAL])
for i, v in enumerate([b["gap"], e["gap"]]):
    a.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=10, color=INK)
a.set_xticks([0, 1]); a.set_xticklabels(["Original\n(10:1)", "Enhanced\n(1:1)"])
a.set_ylim(0, 0.28); a.set_ylabel("|C-index(men) − C-index(women)|")
a.set_title(f"Sex disparity reduced {R['gap_reduction_pct']:.0f}%\n"
            f"(women +{R['female_Cindex_gain']:.3f}, men & overall maintained)")

fig.suptitle("Downstream survival model — bias reduction improves fairness "
             "(illustrative surrogate stress test aligned with planned downstream evaluation)",
             fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(BASE / "poc_survival_fairness.png", bbox_inches="tight")
print("saved poc_survival_fairness.png")
