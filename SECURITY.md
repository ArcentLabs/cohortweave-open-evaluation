# Security Policy

## Reporting a vulnerability

Please do not report security vulnerabilities through public GitHub issues, pull requests, discussions, or comments.

Report suspected vulnerabilities privately to:

**Email:** eva@arcentlabs.com

Please include:

- a clear description of the issue;
- the affected file, component, or workflow;
- steps to reproduce the issue;
- the potential impact;
- any suggested mitigation, if available.

## Sensitive-data policy

Do not include any of the following in a report, issue, pull request, attachment, or code sample:

- protected health information or identifiable patient data;
- clinical DICOM studies or patient-level exports;
- protected EUCAIM data;
- infrastructure credentials, API keys, tokens, or endpoints;
- private model weights or checkpoints;
- partner-confidential material;
- application documents, CVs, or letters of intent.

If sensitive information has already been posted publicly, contact us immediately by email and do not repeat the information in a new public message.

## Scope

This policy applies to the public code, scripts, workflows, fixtures, documentation, and release-audit tooling in this repository.

The repository contains only non-sensitive evaluation and reproducibility components. Protected-data experiments are performed in controlled environments and are outside the scope of public issue reporting.

## Response process

After receiving a report, ArcentLabs will:

1. acknowledge receipt when possible;
2. assess the report and determine whether it is in scope;
3. coordinate a fix or mitigation;
4. communicate with the reporter about disclosure timing when appropriate.

Please allow the maintainers to complete the review before publicly disclosing a suspected vulnerability.
