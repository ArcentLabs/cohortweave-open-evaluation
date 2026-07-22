# Responsible AI and release gates

CohortWeave outputs are research/data-infrastructure artefacts, not replacements for clinical evidence.

Every candidate release should be evaluated for:

- clinical consistency and constraint violations;
- fidelity and correlation preservation;
- subgroup representation, utility and calibration;
- uncertainty and abstention;
- hallucinated/deleted image structures and radiomics drift;
- membership, linkage, singling-out and attribute-inference risk;
- provenance, code/model/data versions and failure reasons.

A strong average metric cannot compensate for a failed safety gate. Protected data and unsafe synthetic outputs remain inside the secure environment.
