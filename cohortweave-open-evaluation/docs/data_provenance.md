# Data provenance

## Included fixtures

- `synthetic_clinical.csv`: 64 procedurally generated rows derived from the documented surrogate generator. Subject IDs are synthetic labels.
- `synthetic_ct_phantom.nii.gz`: 32×32×16 procedurally generated HU-like texture volume. It contains no real anatomy.
- JSON schemas and fixture hashes.

## Frozen results

Aggregate JSON/CSV outputs come from the Spark PoC. The case-level public CT CSV replaces public package Patient IDs with local `PUBLIC_CT_XX` labels. No DICOM pixels are included.

## Public CT proxy

The optional script accesses test fixtures installed with `pydicom`/`pydicom-data`. Source filenames and SHA-256 hashes are recorded by the benchmark. Verify the current third-party package licence before redistribution. This repository does not redistribute the DICOM files.

## Challenge metadata

Only aggregate public-index QC is included. The raw archive and index are excluded. The aggregate does not imply access to protected clinical JSON or complete DICOM studies.

## Prohibited additions

Do not add protected or locally acquired patient data, direct/indirect identifiers, DICOM headers, private cohort statistics below approved disclosure thresholds, partner correspondence, credentials or proprietary checkpoints.
