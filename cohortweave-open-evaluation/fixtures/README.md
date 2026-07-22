# Synthetic fixtures

These files are deliberately small and contain no real patient data.

- `synthetic_clinical.csv`: 64 synthetic clinical rows.
- `synthetic_ct_phantom.nii.gz`: small procedural HU-like NIfTI volume.
- `clinical_fixture.schema.json`: validation schema.
- `fixture_manifest.json`: sizes, provenance and SHA-256 hashes.

They exist so tests and examples can run without GDPR-regulated or licensed clinical datasets.
