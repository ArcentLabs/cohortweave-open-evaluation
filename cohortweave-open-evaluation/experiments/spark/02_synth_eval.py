"""
AI-BOOST Challenge 3 — PoC step 2
Gaussian-copula conditional tabular synthesizer + evaluation against the
official KPIs (1 Fidelity, 2 Bias/Imbalance reduction, 3 Clinical completion).

Method: rank/frequency -> Gaussian latent copula. Marginals are preserved
exactly by construction; the latent correlation matrix captures the
cross-variable dependence structure (stage->metastasis, smoking->histology,
age->ECOG, ...). Conditional/stratified fitting enables targeted augmentation
of underrepresented subgroups. The same architecture generalises to
TVAE / CTGAN / tabular-diffusion for the ADVANCE phase on CINECA.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

BASE = Path(__file__).resolve().parent
RNG = np.random.default_rng(7)

CONT = ["age_at_baseline", "pd_l1"]
ID_COLS = ["subject_id", "date_baseline_ct", "clinical_staging_date", "date_smoking_status", "death_date"]


class GaussianCopula:
    """Minimal, dependency-free Gaussian-copula synthesizer for mixed data."""

    def __init__(self):
        self.cols = None
        self.kinds = {}          # 'cont' or 'cat'
        self.cont_sorted = {}    # sorted values for empirical inverse-CDF
        self.cat_levels = {}     # category -> (lo, hi) cumulative-freq interval
        self.corr = None

    def _to_uniform(self, df):
        U = np.zeros((len(df), len(self.cols)))
        for j, c in enumerate(self.cols):
            x = df[c].to_numpy()
            if self.kinds[c] == "cont":
                # empirical CDF via ranks (ties -> average), mapped to (0,1)
                s = pd.Series(x).rank(method="average").to_numpy()
                U[:, j] = s / (len(x) + 1)
            else:
                lo_hi = self.cat_levels[c]
                u = np.empty(len(x))
                for i, v in enumerate(x):
                    lo, hi = lo_hi[v]
                    u[i] = RNG.uniform(lo, hi)  # spread mass within category band
                U[:, j] = u
        return np.clip(U, 1e-4, 1 - 1e-4)

    def fit(self, df):
        self.cols = list(df.columns)
        for c in self.cols:
            if c in CONT:
                self.kinds[c] = "cont"
                self.cont_sorted[c] = np.sort(df[c].dropna().to_numpy())
            else:
                self.kinds[c] = "cat"
                vc = df[c].astype(str).value_counts(normalize=True)
                bands, cum = {}, 0.0
                for lvl, p in vc.items():
                    bands[lvl] = (cum, cum + p)
                    cum += p
                self.cat_levels[c] = bands
        # work on non-missing rows for correlation estimation
        d = df.copy()
        for c in CONT:
            if c in d:
                d[c] = d[c].fillna(d[c].median())
        d = d.astype({c: str for c in self.cols if self.kinds[c] == "cat"})
        U = self._to_uniform(d)
        Z = norm.ppf(U)
        self.corr = np.corrcoef(Z, rowvar=False)
        # regularise to ensure positive-definiteness
        self.corr += np.eye(len(self.cols)) * 1e-3
        return self

    def sample(self, n):
        L = np.linalg.cholesky(self.corr)
        Z = RNG.standard_normal((n, len(self.cols))) @ L.T
        U = norm.cdf(Z)
        out = {}
        for j, c in enumerate(self.cols):
            u = U[:, j]
            if self.kinds[c] == "cont":
                s = self.cont_sorted[c]
                idx = np.clip((u * (len(s) - 1)).round().astype(int), 0, len(s) - 1)
                out[c] = s[idx]
            else:
                bands = self.cat_levels[c]
                vals = np.empty(n, dtype=object)
                for i, ui in enumerate(u):
                    for lvl, (lo, hi) in bands.items():
                        if lo <= ui < hi:
                            vals[i] = lvl
                            break
                    else:
                        vals[i] = list(bands)[-1]
                out[c] = vals
        return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def kpi1_fidelity(real, synth, cont, cats):
    ks = {}
    for c in cont:
        a, b = real[c].dropna(), synth[c].dropna()
        ks[c] = float(ks_2samp(a, b).statistic)
    prev_diff = {}
    for c in cats:
        pr = real[c].astype(str).value_counts(normalize=True)
        ps = synth[c].astype(str).value_counts(normalize=True)
        levels = set(pr.index) | set(ps.index)
        prev_diff[c] = float(np.mean([abs(pr.get(l, 0) - ps.get(l, 0)) for l in levels]) * 100)
    ks_ok = np.mean([v <= 0.10 for v in ks.values()]) * 100
    prev_ok = np.mean([v <= 5.0 for v in prev_diff.values()]) * 100
    return ks, prev_diff, ks_ok, prev_ok


def imbalance_ratio(series, groups):
    counts = np.array([max(1, (series == g).sum()) for g in groups])
    return counts.max() / counts.min()


def privacy_smoke_test(real, synth, cont, cats):
    """Exact-match replay + distance-to-closest-record (DCR). Smoke tests only -
    not evidence of anonymisation; full membership-inference/linkage/singling-out
    tests run in the secure environment in ADVANCE."""
    # exact replay on categorical + rounded-continuous signature
    def signature(df):
        s = df[cats].astype(str).agg("|".join, axis=1)
        for c in cont:
            s = s + "|" + df[c].round(0).astype("Int64").astype(str)
        return set(s)
    real_sig = signature(real)
    syn_sig = df_sig = synth[cats].astype(str).agg("|".join, axis=1)
    for c in cont:
        df_sig = df_sig + "|" + synth[c].round(0).astype("Int64").astype(str)
    exact = float(np.mean([s in real_sig for s in df_sig]) * 100)
    # DCR on standardised continuous features (synthetic -> nearest real)
    from sklearn.neighbors import NearestNeighbors
    R = real[cont].fillna(real[cont].median()).to_numpy()
    S = synth[cont].fillna(real[cont].median()).to_numpy()
    mu, sd = R.mean(0), R.std(0) + 1e-9
    nn = NearestNeighbors(n_neighbors=1).fit((R - mu) / sd)
    d, _ = nn.kneighbors((S - mu) / sd)
    return {"exact_row_match_rate_pct": round(exact, 3),
            "numeric_dcr_median_standardized": round(float(np.median(d)), 4),
            "numeric_dcr_p05_standardized": round(float(np.quantile(d, 0.05)), 4)}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    real = pd.read_pickle(BASE / "cohort_real.pkl")
    feat = [c for c in real.columns if c not in ID_COLS]
    cats = [c for c in feat if c not in CONT]

    # ---- Fit unconditional synthesizer & KPI-1 fidelity -------------------
    gc = GaussianCopula().fit(real[feat])
    synth = gc.sample(len(real))
    ks, prev_diff, ks_ok, prev_ok = kpi1_fidelity(real, synth, CONT, cats)

    # correlation-structure preservation (extra fidelity signal)
    def corr_num(df):
        num = df.copy()
        for c in cats:
            num[c] = num[c].astype("category").cat.codes
        return num[feat].corr().to_numpy()
    corr_err = float(np.nanmean(np.abs(corr_num(real) - corr_num(synth))))

    # ---- KPI-2: bias reduction via stratified augmentation ----------------
    # Fit gender-specific copulas; augment minority (female) to reach ~parity.
    groups = ["male", "female"]
    ir_before = imbalance_ratio(real["gender"].to_numpy(), groups)
    n_m = (real["gender"] == "male").sum()
    n_f = (real["gender"] == "female").sum()
    gc_f = GaussianCopula().fit(real[real["gender"] == "female"][feat])
    n_needed = n_m - n_f
    synth_f = gc_f.sample(n_needed)
    augmented = pd.concat([real[feat], synth_f], ignore_index=True)
    ir_after = imbalance_ratio(augmented["gender"].to_numpy(), groups)
    ir_reduction = (1 - ir_after / ir_before) * 100
    excess_imbalance_removed = (1 - (ir_after - 1) / (ir_before - 1)) * 100

    # intersectional check: gender x age-group
    def agebin(a):
        return pd.cut(a, [0, 55, 65, 75, 200], labels=["<55", "55-64", "65-74", "75+"])
    real_ix = real["gender"].astype(str) + "|" + agebin(real["age_at_baseline"]).astype(str)
    aug_ix = augmented["gender"].astype(str) + "|" + agebin(augmented["age_at_baseline"]).astype(str)
    strata = sorted(set(real_ix))
    ir_ix_before = imbalance_ratio(real_ix.to_numpy(), strata)
    ir_ix_after = imbalance_ratio(aug_ix.to_numpy(), strata)

    # fidelity is preserved in the female synthetic block
    ks_f = {c: float(ks_2samp(real[real.gender == "female"][c].dropna(),
                              synth_f[c].dropna()).statistic) for c in CONT}

    # ---- KPI-3: clinical completion (mask -> impute vs median/mode) --------
    completion = {}
    cat_targets = ["clinical_stage_group", "ecog_performance_status", "treatment_intent",
                   "metastasis_clinical_category", "distant_metastasis_pr"]
    for tgt in cat_targets:
        d = real[feat].copy()
        y = d[tgt].astype(str)
        X = d.drop(columns=[tgt]).copy()
        for c in X.columns:
            if c in cats:
                X[c] = X[c].astype("category").cat.codes
            else:
                X[c] = X[c].fillna(X[c].median())
        mask = RNG.random(len(d)) < 0.30
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        clf.fit(X[~mask], y[~mask])
        pred = clf.predict(X[mask])
        acc_model = float((pred == y[mask].to_numpy()).mean())
        mode = y[~mask].mode()[0]
        acc_base = float((y[mask] == mode).mean())
        completion[tgt] = {"model_accuracy": round(acc_model, 4),
                           "mode_baseline": round(acc_base, 4)}

    # continuous completion: pd_l1 MAE, model vs median
    d = real[feat].copy()
    obs = d["pd_l1"].notna()
    y = d.loc[obs, "pd_l1"].to_numpy()
    X = d.loc[obs].drop(columns=["pd_l1"]).copy()
    for c in X.columns:
        if c in cats:
            X[c] = X[c].astype("category").cat.codes
        else:
            X[c] = X[c].fillna(X[c].median())
    mask = RNG.random(obs.sum()) < 0.30
    reg = HistGradientBoostingRegressor(max_iter=250, random_state=0)
    reg.fit(X[~mask], y[~mask])
    mae_model = float(np.abs(reg.predict(X[mask]) - y[mask]).mean())
    mae_base = float(np.abs(np.median(y[~mask]) - y[mask]).mean())
    mae_impr = (1 - mae_model / mae_base) * 100

    # ---- privacy smoke test -----------------------------------------------
    privacy = privacy_smoke_test(real, synth, CONT, cats)

    # ---- collect + save ----------------------------------------------------
    import hashlib, sys
    script_hash = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    results = {
        "run_metadata": {"seed_synth": 7, "python": sys.version.split()[0],
                         "script_sha256": script_hash},
        "kpi1_fidelity": {
            "ks_distance": {k: round(v, 4) for k, v in ks.items()},
            "pct_continuous_vars_KS_le_0.10": round(ks_ok, 1),
            "prevalence_abs_diff_pct": {k: round(v, 2) for k, v in prev_diff.items()},
            "pct_categorical_vars_prevdiff_le_5pct": round(prev_ok, 1),
            "mean_abs_correlation_error": round(corr_err, 4),
        },
        "kpi2_bias_reduction": {
            "imbalance_ratio_before": round(float(ir_before), 3),
            "imbalance_ratio_after": round(float(ir_after), 3),
            "imbalance_reduction_pct_gender_official_ratio": round(float(ir_reduction), 1),
            "excess_imbalance_removed_pct": round(float(excess_imbalance_removed), 1),
            "intersectional_gender_x_age_before": round(float(ir_ix_before), 3),
            "intersectional_gender_x_age_after": round(float(ir_ix_after), 3),
            "minority_block_KS": {k: round(v, 4) for k, v in ks_f.items()},
        },
        "kpi3_clinical_completion": {
            "categorical": completion,
            "pd_l1_MAE_model": round(mae_model, 3),
            "pd_l1_MAE_median_baseline": round(mae_base, 3),
            "pd_l1_MAE_improvement_pct": round(float(mae_impr), 1),
            "categorical_targets_meeting_90pct": int(sum(v["model_accuracy"] >= 0.90 for v in completion.values())),
            "categorical_targets_tested": int(len(completion)),
            "imaging_completion_status": "not_implemented_in_spark_poc",
        },
        "privacy_smoke_test": {**privacy,
            "warning": "Smoke tests only; not evidence of anonymisation. "
                       "Membership-inference, linkage and singling-out tests run "
                       "in the secure environment."},
    }
    with open(BASE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    synth.to_pickle(BASE / "cohort_synth.pkl")
    augmented.to_pickle(BASE / "cohort_augmented.pkl")

    print(json.dumps(results, indent=2))
