# Reproduce Spark evidence

## 1. Quick verification (target: under 10 minutes)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest
python scripts/verify_frozen_results.py
python scripts/check_public_release.py
```

Expected terminal output contains only `PASS` lines and a final `PUBLIC RELEASE AUDIT PASSED` message.

## 2. Re-run non-sensitive synthetic experiments

```bash
python scripts/reproduce_spark.py --output artifacts/latest
```

This copies the frozen experiment code into a temporary directory, generates a deterministic 1,088-row surrogate cohort, reruns synthesis/completion, phantom imaging, downstream fairness and constraint ablation, then copies JSON/PNG outputs into `artifacts/latest`. Generated `.pkl` files remain in the temporary directory and are deleted automatically.

## 3. Optional public real-pixel proxy

```bash
python -m pip install -r requirements-real-ct.txt
python scripts/reproduce_spark.py --include-real-ct --output artifacts/with-real-ct
```

The script loads public CT test fixtures distributed by the `pydicom`/`pydicom-data` packages. It does not download or use protected Challenge 3 studies. The benchmark is a restoration proxy with synthetic corruption, not paired protocol translation and not ICR.

## 4. Optional public-index QC

The raw challenge archive and `index.json` are intentionally absent. A permitted user may run:

```bash
python experiments/spark/00_public_index_qc.py --archive /local/path/14046442.zip --output artifacts/index_qc.json
```

Do not commit the archive, raw index or any protected material. Only aggregate QC may be considered for publication after a licence/privacy review.

## Determinism

Scripts use fixed seeds and relative paths. Small floating-point differences may occur across BLAS, CPU and library builds. Frozen release gates use thresholds rather than byte-identical model outputs.
