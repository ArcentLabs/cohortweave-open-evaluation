# Contributing

1. Work only with synthetic fixtures or explicitly redistributable public data.
2. Add tests for every metric or release gate change.
3. Run `pytest`, `python scripts/verify_frozen_results.py` and `python scripts/check_public_release.py`.
4. Do not commit DICOM/NIfTI derived from patients, model weights, pickles, credentials, application documents, CVs or partner letters.
5. Describe evidence boundaries and failure cases in every pull request.
