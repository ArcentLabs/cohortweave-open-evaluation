# Contributing to CohortWeave Open Evaluation

Thank you for your interest in contributing to **CohortWeave Open Evaluation**.

This repository contains public, non-sensitive evaluation and reproducibility components for CohortWeave. Contributions should preserve the repository's evidence boundaries, privacy safeguards, reproducibility requirements, and scientific transparency.

## Before contributing

Please read:

- `README.md`
- `docs/evidence_boundaries.md`
- `docs/data_provenance.md`
- `docs/reproduce_spark.md`
- `docs/public_release_checklist.md`

By contributing, you confirm that your submission does not contain confidential, restricted, or personal data.

## Contributions we welcome

Examples include:

- corrections to documentation;
- improvements to reproducibility instructions;
- additional tests for metric implementations;
- fixes to synthetic-data or phantom workflows;
- clearer validation and release-audit checks;
- performance improvements that preserve the documented outputs;
- reproducible, non-sensitive examples;
- improvements to accessibility and code readability.

## Do not submit

Do not submit or reference:

- protected health information or identifiable patient data;
- protected EUCAIM data;
- clinical DICOM studies or patient-level exports;
- raw challenge archives or private indexes;
- private model weights or checkpoints;
- `.pkl` cohort files containing restricted data;
- infrastructure credentials, API keys, tokens, or private endpoints;
- partner-confidential information;
- application documents, CVs, letters of intent, or internal correspondence;
- claims of clinical validity that are not supported by the repository's evidence boundaries.

When in doubt, contact **eva@arcentlabs.com** before opening a pull request.

## Development setup

```bash
git clone https://github.com/ArcentLabs/cohortweave-open-evaluation.git
cd cohortweave-open-evaluation

python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the checks

Before submitting a pull request, run:

```bash
pytest
python scripts/verify_frozen_results.py
python scripts/check_public_release.py
```

For the full synthetic Spark workflow:

```bash
python scripts/reproduce_spark.py
```

Do not modify frozen results merely to make checks pass. If a change legitimately updates a frozen result, explain why in the pull request and include the relevant reproducibility evidence.

## Opening an issue

Use an issue for:

- a reproducible bug;
- a documentation problem;
- a proposed non-sensitive enhancement;
- a question about the public evaluation workflow.

Do not use a public issue to report a security vulnerability or to share sensitive data. Follow `SECURITY.md` instead.

## Pull-request guidelines

A pull request should:

1. have a clear, focused purpose;
2. describe the problem and the proposed change;
3. include or update tests when relevant;
4. pass the repository checks;
5. update documentation when behavior changes;
6. preserve the documented evidence boundaries;
7. avoid unrelated formatting or refactoring changes.

Please keep pull requests small enough to review efficiently.

## Scientific and claims policy

Contributions must distinguish clearly between:

- demonstrated public results;
- surrogate or synthetic evaluations;
- public proxy evaluations;
- planned or future work;
- clinical claims.

Do not imply that this repository demonstrates clinical validity, paired same-patient missing-acquisition completion, or achievement of the official ICR target unless the evidence and repository documentation have been updated to support that conclusion.

## Code style

Follow the existing project structure and style. Prefer:

- clear function names;
- explicit error handling;
- deterministic behavior where possible;
- concise comments for non-obvious logic;
- tests for new behavior;
- no unnecessary dependencies.

## Licence

By submitting a contribution, you agree that your contribution will be made available under the repository's **Apache License 2.0**.
