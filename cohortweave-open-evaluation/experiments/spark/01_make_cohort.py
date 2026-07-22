"""
AI-BOOST Challenge 3 — PoC step 1
Schema-complete surrogate NSCLC cohort for engineering validation.

The real EUCAIM/CHAIMELEON data (1088 thorax-CT subjects) is only accessible
inside the Secure Processing Environment during the ADVANCE phase. For the SPARK
concept-note PoC we therefore build a surrogate cohort that implements the 29-field JSON schema published in the Challenge Description and realistic, *correlated*
clinical distributions for advanced non-small-cell lung cancer (NSCLC). This lets
us validate the full synthesis + evaluation pipeline and report genuine metrics
against the official KPIs. Every distribution below is a modelling assumption,
clearly documented; no real patient data is used.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

RNG = np.random.default_rng(42)
N = 1088  # match the published cohort size


def choice(options, probs, size):
    return RNG.choice(options, size=size, p=np.array(probs) / np.sum(probs))


def make_cohort(n=N):
    # --- demographics -------------------------------------------------------
    # Age: advanced NSCLC, right-skewed around mid-60s, clipped to 29-91.
    age = np.clip(RNG.normal(66, 10, n), 29, 91).round(0)

    # Gender: deliberately imbalanced (~68% M / 32% F) to reflect real NSCLC
    # skew AND to give KPI-2 (bias/imbalance reduction) a meaningful target.
    gender = choice(["male", "female"], [0.68, 0.32], n)

    # Smoking status correlated with gender (recoverable signal): males skew
    # former/current, females carry most never-smokers (real NSCLC pattern).
    smoking_status = np.empty(n, dtype=object)
    for i in range(n):
        older = age[i] >= 65
        if gender[i] == "male":
            p = [0.70, 0.25, 0.05] if older else [0.45, 0.48, 0.07]
        else:
            p = [0.42, 0.13, 0.45] if older else [0.30, 0.28, 0.42]
        smoking_status[i] = choice(["former", "current", "never"], p, 1)[0]

    # --- tumour biology -----------------------------------------------------
    # Histotype depends on smoking: never-smokers skew adenocarcinoma.
    histo = np.empty(n, dtype=object)
    for i in range(n):
        if smoking_status[i] == "never":
            histo[i] = choice(["adenocarcinoma", "squamous_cell_carcinoma",
                               "large_cell_carcinoma", "nsclc_nos"],
                              [0.72, 0.10, 0.08, 0.10], 1)[0]
        else:
            histo[i] = choice(["adenocarcinoma", "squamous_cell_carcinoma",
                               "large_cell_carcinoma", "nsclc_nos"],
                              [0.50, 0.34, 0.08, 0.08], 1)[0]

    # Clinical stage group (advanced cohort: mostly III/IV).
    stage = choice(["I", "II", "III", "IV"], [0.10, 0.14, 0.34, 0.42], n)
    stage_idx = np.array({"I": 0, "II": 1, "III": 2, "IV": 3}[s] for s in stage) \
        if False else np.array([{"I": 0, "II": 1, "III": 2, "IV": 3}[s] for s in stage])

    # T / N categories loosely increasing with stage.
    T = np.empty(n, dtype=object)
    Ncat = np.empty(n, dtype=object)
    for i in range(n):
        si = stage_idx[i]
        T[i] = choice(["T1", "T2", "T3", "T4"],
                      [max(0.05, 0.4 - 0.1 * si), 0.35, 0.2 + 0.05 * si, 0.1 + 0.07 * si], 1)[0]
        Ncat[i] = choice(["N0", "N1", "N2", "N3"],
                         [max(0.05, 0.5 - 0.13 * si), 0.25, 0.15 + 0.06 * si, 0.1 + 0.07 * si], 1)[0]

    # --- metastases: driven by stage (only stage IV carries high M burden) --
    def met_prob(base_iv):
        # For the proof-of-mechanism cohort we make the TNM logic explicit:
        # distant metastatic sites occur only in stage IV. This creates an
        # internally coherent surrogate ground truth for testing whether a
        # generative model re-introduces contradictions.
        return {"I": 0.0, "II": 0.0, "III": 0.0, "IV": base_iv}

    met_sites = {
        "metastasis_lung":        met_prob(0.30),
        "metastasis_pleura":      met_prob(0.22),
        "metastasis_lymph_nodes": met_prob(0.55),
        "metastasis_adrenal_gland": met_prob(0.18),
        "metastasis_liver":       met_prob(0.20),
        "metastasis_brain":       met_prob(0.25),
        "metastasis_bone":        met_prob(0.30),
        "metastasis_other":       met_prob(0.12),
    }
    mets = {}
    for name, pmap in met_sites.items():
        mets[name] = np.array([1 if RNG.random() < pmap[stage[i]] else 0 for i in range(n)])

    # Ensure every stage-IV patient has at least one recorded metastatic site.
    stage_iv_idx = np.where(stage == "IV")[0]
    site_names = list(mets)
    site_weights = np.array([0.30, 0.22, 0.55, 0.18, 0.20, 0.25, 0.30, 0.12], dtype=float)
    site_weights /= site_weights.sum()
    for i in stage_iv_idx:
        if not any(mets[name][i] for name in site_names):
            mets[RNG.choice(site_names, p=site_weights)][i] = 1

    any_distant = np.zeros(n, dtype=int)
    for name in mets:
        any_distant = np.maximum(any_distant, mets[name])

    # M clinical category derived from the metastatic pattern, rather than
    # sampled independently. M0 is reserved for non-stage-IV patients.
    m_cat = np.full(n, "M0", dtype=object)
    extra_sites = ["metastasis_adrenal_gland", "metastasis_liver",
                   "metastasis_brain", "metastasis_bone", "metastasis_other"]
    for i in stage_iv_idx:
        n_extra = sum(int(mets[name][i]) for name in extra_sites)
        m_cat[i] = "M1c" if n_extra >= 2 else ("M1b" if n_extra == 1 else "M1a")

    # --- functional status & treatment -------------------------------------
    # ECOG worsens with age and stage (tight, recoverable relationship).
    ecog = np.clip(np.round(
        0.55 * (stage_idx) + 0.05 * (age - 60) + RNG.normal(0, 0.45, n)
    ), 0, 4).astype(int)

    treatment_intent = np.where(
        (stage == "IV") | (ecog >= 2),
        choice(["palliative", "curative"], [0.8, 0.2], n),
        choice(["curative", "palliative"], [0.75, 0.25], n),
    )

    # --- biomarkers ---------------------------------------------------------
    # PD-L1 tumour proportion score (%): heavy mass at 0, spread 0-100.
    pd_l1_unknown = (RNG.random(n) < 0.18).astype(int)
    # PD-L1 TPS associates with histology (squamous higher) and smoking status.
    histo_shift = np.array([{"squamous_cell_carcinoma": 34, "adenocarcinoma": 0,
                             "large_cell_carcinoma": 16, "nsclc_nos": 8}[h] for h in histo])
    smoke_shift = np.array([{"current": 18, "former": 9, "never": 0}[s] for s in smoking_status])
    pd_l1_raw = np.clip(RNG.beta(0.7, 1.6, n) * 55 + histo_shift + smoke_shift, 0, 100)
    pd_l1 = np.where(pd_l1_unknown == 1, np.nan, pd_l1_raw.round(0))

    # --- outcomes -----------------------------------------------------------
    prog = np.array([1 if RNG.random() < (0.2 + 0.13 * stage_idx[i]) else 0 for i in range(n)])
    local_rec = np.array([1 if RNG.random() < (0.1 + 0.05 * stage_idx[i]) else 0 for i in range(n)])
    death_cancer = np.array([1 if RNG.random() < (0.12 + 0.15 * stage_idx[i]) else 0 for i in range(n)])

    df = pd.DataFrame({
        "subject_id": [f"SUBJ_{i:04d}" for i in range(n)],
        "date_baseline_ct": pd.to_datetime("2019-01-01") + pd.to_timedelta(RNG.integers(0, 1200, n), "D"),
        "age_at_baseline": age.astype(int),
        "gender": gender,
        "tumor_histotype": histo,
        "pd_l1": pd_l1,
        "pd_l1_unknown": pd_l1_unknown,
        "local_recurrence_progression": local_rec,
        "death_related_to_cancer": death_cancer,
        "clinical_stage_group": stage,
        **mets,
        "treatment_intent": treatment_intent,
        "smoking_status": smoking_status,
        "ecog_performance_status": ecog,
        "metastasis_clinical_category": m_cat,
        "progression_recurrence": prog,
        "distant_metastasis_pr": any_distant,
        "tumor_clinical_category": T,
        "regional_nodes_clinical_category": Ncat,
    })
    # Complete the published 29-field schema. The current PoC does not model
    # date variables; they are retained for ingestion/schema coverage and are
    # explicitly excluded from the synthesis smoke test.
    df["clinical_staging_date"] = df["date_baseline_ct"] - pd.to_timedelta(RNG.integers(0, 31, n), "D")
    df["date_smoking_status"] = df["date_baseline_ct"] - pd.to_timedelta(RNG.integers(0, 366, n), "D")
    death_delay = pd.to_timedelta(RNG.integers(30, 1500, n), "D")
    df["death_date"] = pd.NaT
    died = df["death_related_to_cancer"].astype(bool)
    df.loc[died, "death_date"] = df.loc[died, "date_baseline_ct"] + death_delay[died]
    official_order = [
        "subject_id", "date_baseline_ct", "age_at_baseline", "gender", "tumor_histotype",
        "pd_l1", "pd_l1_unknown", "local_recurrence_progression", "death_related_to_cancer",
        "clinical_staging_date", "clinical_stage_group", "metastasis_lung", "metastasis_pleura",
        "metastasis_lymph_nodes", "metastasis_adrenal_gland", "metastasis_liver",
        "metastasis_brain", "metastasis_bone", "metastasis_other", "treatment_intent",
        "smoking_status", "date_smoking_status", "ecog_performance_status",
        "metastasis_clinical_category", "progression_recurrence", "distant_metastasis_pr",
        "tumor_clinical_category", "regional_nodes_clinical_category", "death_date",
    ]
    return df[official_order]


if __name__ == "__main__":
    df = make_cohort()
    df.to_pickle(BASE / "cohort_real.pkl")
    print(f"Surrogate cohort: {df.shape[0]} subjects x {df.shape[1]} variables")
    print("\nGender balance (drives KPI-2):")
    print(df["gender"].value_counts())
    print("\nStage distribution:")
    print(df["clinical_stage_group"].value_counts().sort_index())
    print("\nSample rows:")
    print(df[["age_at_baseline", "gender", "tumor_histotype", "clinical_stage_group",
              "pd_l1", "ecog_performance_status", "distant_metastasis_pr"]].head())
    print("\nMissing pd_l1 (unknown):", int(df["pd_l1"].isna().sum()))
