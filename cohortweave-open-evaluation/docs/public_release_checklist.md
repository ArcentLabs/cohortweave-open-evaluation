# Public release checklist

Before the first GitHub release:

- [ ] Confirm the repository owner is the legal applicant or approved project organisation.
- [ ] Review every team/affiliation statement separately from this technical repository.
- [ ] Run `pytest`.
- [ ] Run `python scripts/verify_frozen_results.py`.
- [ ] Run `python scripts/check_public_release.py`.
- [ ] Inspect git history for deleted secrets or medical files.
- [ ] Confirm no DICOM, raw challenge archive/index, `.pkl`, weights, CV, LoI or application document is present.
- [ ] Confirm third-party licences for optional public fixtures.
- [ ] Create tag `v0.1.0-spark` and GitHub release.
- [ ] Connect the public repository to Zenodo and archive the release.
- [ ] Add the final GitHub URL and DOI to the application Open Science field.

Use `python scripts/make_release_archive.py` to create an audited upload ZIP.
