"""AI-BOOST Challenge 3 - Spark proof-of-mechanism constraint ablation.

This experiment does NOT claim clinical validation. It uses the deterministic,
schema-aligned surrogate cohort because protected clinical JSON and DICOM pixels
are unavailable before the ADVANCE secure-environment phase.

The purpose is narrower and falsifiable: test whether explicit patient-level
clinical constraints add measurable value beyond an unconstrained mixed-type
generator or post-hoc filtering.

Variants
--------
A. Unconstrained Gaussian-copula generation.
B. Post-hoc filtering: reject any row violating a pre-specified rule.
C. Constraint-aware repair proxy: enforce the rules before release.
D. Fairness steering + constraint-aware repair + an evidence gate.

The repair is a transparent proof-of-mechanism proxy, not the final ADVANCE
constraint-guided diffusion implementation. ADVANCE will compare learned and
clinician-authored guidance against these baselines on protected data.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SEED = 20260716
MET_COLS = [
    "metastasis_lung", "metastasis_pleura", "metastasis_lymph_nodes",
    "metastasis_adrenal_gland", "metastasis_liver", "metastasis_brain",
    "metastasis_bone", "metastasis_other",
]
EXTRA_SITES = [
    "metastasis_adrenal_gland", "metastasis_liver", "metastasis_brain",
    "metastasis_bone", "metastasis_other",
]


def load_synth_module():
    spec = importlib.util.spec_from_file_location("cw_synth_eval", BASE / "02_synth_eval.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def clinical_rule_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return row-level violations for four explicit logical constraints.

    These are proof-of-mechanism rules derived from the surrogate data contract,
    not a complete clinical ontology. The clinician-reviewed ADVANCE graph will
    be broader and validated on the protected cohort.
    """
    any_site = df[MET_COLS].astype(int).max(axis=1).astype(int)
    distant = df["distant_metastasis_pr"].astype(int)
    m0 = df["metastasis_clinical_category"].astype(str).eq("M0")
    stage4 = df["clinical_stage_group"].astype(str).eq("IV")
    unknown = df["pd_l1_unknown"].astype(str).isin(["1", "1.0", "True", "true"])
    missing = df["pd_l1"].isna()

    return pd.DataFrame({
        "distant_flag_matches_recorded_sites": distant.ne(any_site),
        "m_category_matches_distant_flag": (m0 & distant.eq(1)) | ((~m0) & distant.eq(0)),
        "stage_iv_matches_m1": stage4.ne(~m0),
        "pdl1_unknown_matches_missingness": unknown.ne(missing),
    }, index=df.index)


def m_category_from_sites(row: pd.Series) -> str:
    n_extra = sum(int(row[c]) for c in EXTRA_SITES)
    if n_extra >= 2:
        return "M1c"
    if n_extra == 1:
        return "M1b"
    return "M1a"


def constraint_repair(df: pd.DataFrame, real: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Transparent rule-based proxy for constraint-guided generation.

    Clinical stage is used as the anchor for the surrogate proof of mechanism.
    The final implementation will use probabilistic/clinician-authored guidance,
    uncertainty and abstention rather than unconditional deterministic repair.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)

    stage4 = out["clinical_stage_group"].astype(str).eq("IV")
    non_stage4 = ~stage4

    # Non-stage-IV records: remove distant disease and set M0.
    out.loc[non_stage4, MET_COLS] = 0
    out.loc[non_stage4, "distant_metastasis_pr"] = 0
    out.loc[non_stage4, "metastasis_clinical_category"] = "M0"

    # Stage IV records: require at least one site, distant flag and an M1 class.
    site_probs = real.loc[real["clinical_stage_group"].astype(str).eq("IV"), MET_COLS].mean().to_numpy(float)
    site_probs = site_probs / site_probs.sum()
    out.loc[stage4, "distant_metastasis_pr"] = 1
    for idx in out.index[stage4]:
        if int(out.loc[idx, MET_COLS].astype(int).max()) == 0:
            chosen = rng.choice(MET_COLS, p=site_probs)
            out.loc[idx, chosen] = 1
        out.loc[idx, "metastasis_clinical_category"] = m_category_from_sites(out.loc[idx])

    # PD-L1 missingness must agree with its explicit unknown indicator.
    unknown = out["pd_l1_unknown"].astype(str).isin(["1", "1.0", "True", "true"])
    out.loc[unknown, "pd_l1"] = np.nan
    known_missing = (~unknown) & out["pd_l1"].isna()
    if known_missing.any():
        out.loc[known_missing, "pd_l1"] = float(real["pd_l1"].median())
    return out


def sample_posthoc(gc, n: int, max_draws: int = 50000, batch: int = 2048):
    accepted = []
    total_draws = 0
    while sum(len(x) for x in accepted) < n and total_draws < max_draws:
        x = gc.sample(batch)
        total_draws += len(x)
        valid = ~clinical_rule_matrix(x).any(axis=1)
        accepted.append(x.loc[valid])
    valid_all = pd.concat(accepted, ignore_index=True)
    if len(valid_all) < n:
        raise RuntimeError(f"Only {len(valid_all)} valid rows after {total_draws} draws")
    return valid_all.iloc[:n].copy(), total_draws, len(valid_all)


def imbalance_ratio(series: pd.Series) -> float:
    counts = series.astype(str).value_counts()
    return float(counts.max() / counts.min())


def evaluate_variant(name, df, real, mod, cont, cats, total_draws=None, total_valid=None):
    rules = clinical_rule_matrix(df)
    ks, prev, ks_ok, prev_ok = mod.kpi1_fidelity(real, df, cont, cats)
    non_sensitive = [c for c in cats if c != "gender"]
    _, prev_ns, _, prev_ns_ok = mod.kpi1_fidelity(real, df, cont, non_sensitive)
    privacy = mod.privacy_smoke_test(real, df, cont, cats)
    result = {
        "variant": name,
        "rows": int(len(df)),
        "clinical_contradiction_rate_pct": round(float(rules.any(axis=1).mean() * 100), 2),
        "mean_rule_violations_per_row": round(float(rules.sum(axis=1).mean()), 3),
        "per_rule_violation_pct": {c: round(float(rules[c].mean() * 100), 2) for c in rules},
        "continuous_vars_passing_KS_le_0.10_pct": round(float(ks_ok), 1),
        "categorical_vars_passing_prevalence_le_5pct_pct": round(float(prev_ok), 1),
        "non_sensitive_categorical_vars_passing_prevalence_le_5pct_pct": round(float(prev_ns_ok), 1),
        "mean_abs_categorical_prevalence_diff_pct": round(float(np.mean(list(prev.values()))), 3),
        "mean_abs_non_sensitive_prevalence_diff_pct": round(float(np.mean(list(prev_ns.values()))), 3),
        "max_prevalence_drift": {
            "variable": max(prev, key=prev.get),
            "absolute_difference_pct": round(float(max(prev.values())), 3),
        },
        "imbalance_ratio_male_to_female": round(imbalance_ratio(df["gender"]), 3),
        "ks_distance": {k: round(float(v), 4) for k, v in ks.items()},
        "privacy_smoke_test": privacy,
    }
    if total_draws is not None:
        result["posthoc_total_draws"] = int(total_draws)
        result["posthoc_valid_rows_before_truncation"] = int(total_valid)
        result["posthoc_valid_sample_yield_pct"] = round(float(total_valid / total_draws * 100), 2)
    return result


def evidence_gate(result: dict) -> dict:
    checks = {
        "clinical_contradiction_rate_le_5pct": result["clinical_contradiction_rate_pct"] <= 5.0,
        "continuous_fidelity_80pct_variables": result["continuous_vars_passing_KS_le_0.10_pct"] >= 80.0,
        "non_sensitive_categorical_fidelity_80pct_variables": result["non_sensitive_categorical_vars_passing_prevalence_le_5pct_pct"] >= 80.0,
        "gender_imbalance_ratio_le_1.05": result["imbalance_ratio_male_to_female"] <= 1.05,
        "exact_replay_rate_zero": result["privacy_smoke_test"]["exact_row_match_rate_pct"] == 0.0,
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def main():
    mod = load_synth_module()
    real_full = pd.read_pickle(BASE / "cohort_real.pkl")
    raw = pd.read_pickle(BASE / "cohort_synth.pkl")
    feat = list(raw.columns)
    real = real_full[feat].copy()
    cont = list(mod.CONT)
    cats = [c for c in feat if c not in cont]

    # Confirm the surrogate reference itself satisfies the four test rules.
    reference_rules = clinical_rule_matrix(real)

    # B: post-hoc filtering.
    mod.RNG = np.random.default_rng(71)
    gc = mod.GaussianCopula().fit(real)
    posthoc, total_draws, total_valid = sample_posthoc(gc, len(real))

    # C: transparent constraint-aware repair proxy.
    repaired = constraint_repair(raw, real, SEED)

    # D: gender-parity steering + repair. Gender-specific models preserve
    # subgroup-conditional structure while intentionally changing the sensitive
    # marginal from the original imbalanced cohort to 1:1.
    target_each = len(real) // 2
    mod.RNG = np.random.default_rng(91)
    gc_m = mod.GaussianCopula().fit(real.loc[real["gender"] == "male"])
    male = gc_m.sample(target_each)
    male["gender"] = "male"
    mod.RNG = np.random.default_rng(92)
    gc_f = mod.GaussianCopula().fit(real.loc[real["gender"] == "female"])
    female = gc_f.sample(len(real) - target_each)
    female["gender"] = "female"
    fair_raw = pd.concat([male, female], ignore_index=True)
    fair = constraint_repair(fair_raw, real, SEED + 1)

    variants = {
        "A_unconstrained": raw,
        "B_posthoc_filter": posthoc,
        "C_constraint_repair_proxy": repaired,
        "D_fairness_plus_constraints": fair,
    }
    import hashlib, sys
    results = {
        "run_metadata": {
            "seed": SEED,
            "python": sys.version.split()[0],
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "scope_statement": (
            "Proof-of-mechanism on a deterministic schema-aligned surrogate. "
            "It does not constitute protected-data, clinical or multimodal validation."
        ),
        "reference_surrogate": {
            "rows": int(len(real)),
            "rule_violation_rate_pct": round(float(reference_rules.any(axis=1).mean() * 100), 2),
            "rules_tested": list(reference_rules.columns),
        },
        "variants": {},
    }
    for name, df in variants.items():
        kwargs = {}
        if name == "B_posthoc_filter":
            kwargs = {"total_draws": total_draws, "total_valid": total_valid}
        results["variants"][name] = evaluate_variant(name, df, real, mod, cont, cats, **kwargs)

    results["variants"]["D_fairness_plus_constraints"]["evidence_gate"] = evidence_gate(
        results["variants"]["D_fairness_plus_constraints"]
    )

    # Headline comparisons for application text.
    a = results["variants"]["A_unconstrained"]
    b = results["variants"]["B_posthoc_filter"]
    c = results["variants"]["C_constraint_repair_proxy"]
    d = results["variants"]["D_fairness_plus_constraints"]
    results["headline"] = {
        "raw_contradiction_rate_pct": a["clinical_contradiction_rate_pct"],
        "constraint_proxy_contradiction_rate_pct": c["clinical_contradiction_rate_pct"],
        "relative_contradiction_reduction_pct": round(
            100 * (a["clinical_contradiction_rate_pct"] - c["clinical_contradiction_rate_pct"])
            / max(a["clinical_contradiction_rate_pct"], 1e-9), 1
        ),
        "posthoc_valid_yield_pct": b["posthoc_valid_sample_yield_pct"],
        "constraint_proxy_categorical_fidelity_pass_pct": c["categorical_vars_passing_prevalence_le_5pct_pct"],
        "fair_variant_imbalance_ratio": d["imbalance_ratio_male_to_female"],
        "fair_variant_non_sensitive_fidelity_pass_pct": d["non_sensitive_categorical_vars_passing_prevalence_le_5pct_pct"],
        "fair_variant_evidence_gate_passed": d["evidence_gate"]["passed"],
    }

    with open(BASE / "results_constraint_ablation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    repaired.to_pickle(BASE / "cohort_synth_constraint_repaired.pkl")
    fair.to_pickle(BASE / "cohort_synth_fair_constraint_repaired.pkl")

    # One compact figure for the application appendix.
    labels = ["Unconstrained", "Post-hoc filter", "Constraint proxy", "Fair + constraints"]
    keys = list(variants)
    contradiction = [results["variants"][k]["clinical_contradiction_rate_pct"] for k in keys]
    fidelity = [results["variants"][k]["non_sensitive_categorical_vars_passing_prevalence_le_5pct_pct"] for k in keys]
    imbalance = [results["variants"][k]["imbalance_ratio_male_to_female"] for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].bar(labels, contradiction)
    axes[0].set_title("Rows with any rule violation")
    axes[0].set_ylabel("Percent")
    axes[0].tick_params(axis="x", rotation=25)
    for i, v in enumerate(contradiction):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8)

    axes[1].bar(labels, fidelity)
    axes[1].axhline(80, linestyle="--", linewidth=1)
    axes[1].set_title("Non-sensitive categorical fidelity")
    axes[1].set_ylabel("Variables passing <=5% drift (%)")
    axes[1].set_ylim(0, 110)
    axes[1].tick_params(axis="x", rotation=25)
    for i, v in enumerate(fidelity):
        axes[1].text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=8)

    axes[2].bar(labels, imbalance)
    axes[2].axhline(1.0, linestyle="--", linewidth=1)
    axes[2].set_title("Gender imbalance ratio")
    axes[2].set_ylabel("Larger / smaller group")
    axes[2].tick_params(axis="x", rotation=25)
    for i, v in enumerate(imbalance):
        axes[2].text(i, v + 0.04, f"{v:.2f}", ha="center", fontsize=8)

    fig.suptitle("CohortWeave Spark proof-of-mechanism: constraint ablation on surrogate data")
    fig.tight_layout()
    fig.savefig(BASE / "poc_constraint_ablation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(results["headline"], indent=2))


if __name__ == "__main__":
    main()
