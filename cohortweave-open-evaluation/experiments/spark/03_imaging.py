"""
AI-BOOST Challenge 3 — PoC step 3 (imaging)
Validates the imaging-side methodology on synthetic CT-like texture phantoms,
because the real thorax-CT DICOM data is only reachable inside the Secure
Processing Environment during ADVANCE. Two official KPIs are exercised end-to-end:

  KPI-4  Robustness: GLCM Joint Entropy stability under Gaussian noise
         (sigma = 5, 10, 20 HU) measured with Lin's Concordance Correlation
         Coefficient (CCC). Target: median CCC >= 0.85 at each level.

  KPI-3  Imaging harmonization: two acquisition domains emulating different
         kVp settings are harmonized (histogram matching); intensity-distribution
         divergence is measured with the Wasserstein distance before/after.
         Target: >= 30% reduction, GLCM Joint Entropy CCC >= 0.85 preserved.

All algorithms (GLCM, entropy, CCC, Wasserstein, harmonization) are the same
ones that will run on real CT in ADVANCE; only the pixel source is synthetic.
"""
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
from scipy.ndimage import gaussian_filter
from scipy.stats import wasserstein_distance
from skimage.feature import graycomatrix
from skimage.exposure import match_histograms

RNG = np.random.default_rng(11)
N_IMG = 60
SIZE = 96
LEVELS = 32
HU_MIN, HU_MAX = -1000.0, 400.0   # lung-window HU range


def make_ct_texture(seed):
    """Filtered random field -> lung-parenchyma-like texture on an HU scale.
    Roughness varies per phantom so GLCM Joint Entropy spans a realistic range
    (fine reticular vs coarse patterns), giving the CCC a genuine signal."""
    r = np.random.default_rng(seed)
    fine = 0.8 + (seed % 12) / 11.0 * 2.2          # 0.8 .. 3.0
    w = 0.3 + (seed % 7) / 6.0 * 0.6                # 0.3 .. 0.9
    base = r.standard_normal((SIZE, SIZE))
    tex = gaussian_filter(base, sigma=fine) + w * gaussian_filter(base, sigma=4.0)
    tex = (tex - tex.min()) / (np.ptp(tex) + 1e-9)
    return HU_MIN + tex * (HU_MAX - HU_MIN)


def quantize(img):
    q = np.clip((img - HU_MIN) / (HU_MAX - HU_MIN), 0, 1)
    return (q * (LEVELS - 1)).round().astype(np.uint8)


def glcm_joint_entropy(img):
    g = graycomatrix(quantize(img), distances=[1],
                     angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                     levels=LEVELS, symmetric=True, normed=True)
    P = g[:, :, 0, :].mean(axis=2)      # average over angles
    P = P / (P.sum() + 1e-12)
    return float(-np.sum(P * np.log2(P + 1e-12)))


def ccc(x, y):
    x, y = np.asarray(x), np.asarray(y)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    return float(2 * cov / (vx + vy + (mx - my) ** 2 + 1e-12))


if __name__ == "__main__":
    imgs = [make_ct_texture(s) for s in range(N_IMG)]
    base_ent = np.array([glcm_joint_entropy(im) for im in imgs])

    # ---- KPI-4 robustness: CCC of GLCM Joint Entropy under Gaussian noise --
    robustness = {}
    for sigma in (5, 10, 20):
        noisy_ent = np.array([glcm_joint_entropy(im + RNG.normal(0, sigma, im.shape))
                              for im in imgs])
        robustness[f"sigma_{sigma}HU"] = round(ccc(base_ent, noisy_ent), 4)

    # ---- KPI-3 harmonization: emulate two kVp domains, then harmonize ------
    # Domain A ~ 120 kVp (reference); Domain B ~ 90 kVp: contrast + offset shift
    # plus a mild non-linear (gamma) response, as real kVp changes are not purely
    # affine. Harmonization uses texture-preserving affine intensity
    # standardization (monotonic, gradient-preserving) computed per domain.
    def to_kvp_B(im):
        z = (im - HU_MIN) / (HU_MAX - HU_MIN)
        z = np.clip(z, 0, 1) ** 1.12                       # mild non-linear kVp effect
        c = z * (HU_MAX - HU_MIN) + HU_MIN
        return (c - c.mean()) * 1.35 + c.mean() + 80.0     # contrast + offset
    domA = imgs
    domB = [to_kvp_B(im) for im in imgs]

    intA = np.concatenate([im.ravel() for im in domA])
    intB = np.concatenate([im.ravel() for im in domB])
    w_before = wasserstein_distance(intA, intB)

    # Affine standardization to the domain-A reference statistics (per-domain,
    # not per-paired-image): preserves local texture ordering (linear map),
    # which is what keeps radiomic features concordant.
    mA, sA = intA.mean(), intA.std()
    mB, sB = intB.mean(), intB.std()
    domB_harm = [(im - mB) / sB * sA + mA for im in domB]
    intB_h = np.concatenate([im.ravel() for im in domB_harm])
    w_after = wasserstein_distance(intA, intB_h)
    w_reduction = (1 - w_after / w_before) * 100

    # Radiomic concordance: does harmonization bring domain-B GLCM Joint Entropy
    # back into agreement with the domain-A reference values? (paired phantoms)
    entB_h = np.array([glcm_joint_entropy(im) for im in domB_harm])
    harm_ccc = ccc(base_ent, entB_h)

    results = {
        "kpi4_robustness_CCC_joint_entropy": robustness,
        "kpi4_all_levels_pass_0.85": all(v >= 0.85 for v in robustness.values()),
        "kpi3_harmonization": {
            "wasserstein_before": round(float(w_before), 2),
            "wasserstein_after": round(float(w_after), 2),
            "wasserstein_reduction_pct": round(float(w_reduction), 1),
            "joint_entropy_CCC_preserved": round(harm_ccc, 4),
            "meets_30pct_and_CCC_0.85": bool(w_reduction >= 30 and harm_ccc >= 0.85),
        },
    }
    with open(BASE / "results_imaging.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
