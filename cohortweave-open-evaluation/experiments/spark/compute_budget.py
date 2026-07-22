"""
CohortWeave — bottom-up CINECA Leonardo compute budget.
Planning envelope for 1,088 baseline thorax-CT studies. The public Zenodo index
contains 6,169 indexed series and 142 distinct study-name strings, but these are not
assumed to be 142 acquisition protocols or the final secure-environment workload. A100-hours are single-GPU-equivalent
compute; jobs run multi-GPU (4x A100 / Leonardo Booster node) for wall-clock.
"""

N_STUDIES = 1088
N_SERIES = 6169
N_STUDY_NAME_STRINGS = 142

# (name, params, A100-h per training run, #runs incl. HPO/ablation/final/retrain)
PLAN = [
    ("Tabular diffusion (TabDDPM) + imputers", "5-20M",   2,  30),
    ("2.5D CT autoencoder (latent space)",     "60-120M", 40, 10),
    ("CT latent-diffusion synthesis+completion","150-400M",60, 18),
    ("kVp harmonisation diffusion (+radiomic)", "50-120M", 45, 12),
]

# fixed-cost buckets (A100-h)
EVAL = 350   # inference, KPI eval, 5 monthly leaderboard cycles + finals
PREP = 150   # provisional DICOM conversion / segmentation GPU share; revise after secure inventory
CONTINGENCY = 0.25

print(f"Public-index context: {N_STUDIES} studies, {N_SERIES} indexed series, "
      f"{N_STUDY_NAME_STRINGS} distinct study-name strings (not validated protocols).\n")
print(f"{'Component':44s} {'params':9s} {'h/run':>6s} {'runs':>5s} {'A100-h':>8s}")
print("-" * 78)
train_total = 0
for name, params, hpr, runs in PLAN:
    sub = hpr * runs
    train_total += sub
    print(f"{name:44s} {params:9s} {hpr:6d} {runs:5d} {sub:8d}")
print("-" * 78)
print(f"{'Training subtotal':44s} {'':9s} {'':>6s} {'':>5s} {train_total:8d}")
print(f"{'Evaluation + monthly leaderboard (5 cycles)':44s} {'':9s} {'':>6s} {'':>5s} {EVAL:8d}")
print(f"{'Preprocessing / segmentation (GPU share)':44s} {'':9s} {'':>6s} {'':>5s} {PREP:8d}")
nominal = train_total + EVAL + PREP
cont = round(nominal * CONTINGENCY)
print("-" * 78)
print(f"{'Nominal total':44s} {'':9s} {'':>6s} {'':>5s} {nominal:8d}")
print(f"{'+ contingency (25%)':44s} {'':9s} {'':>6s} {'':>5s} {cont:8d}")
print(f"{'REQUESTED ALLOCATION (rounded)':44s} {'':9s} {'':>6s} {'':>5s} {round((nominal+cont)/500)*500:8d}")
print(f"\nNode-equivalent: {round((nominal+cont)/4)} node-hours "
      f"(~{round((nominal+cont)/4/24)} node-days) on 4x A100 Booster nodes over 5 months.")
print("CPU: ~5,000 core-hours provisional (DICOM conversion, QC and radiomics; revise after secure inventory).")
print("Storage: up to 5 TB provisional scratch; source data remain under the secure environment's governance.")
