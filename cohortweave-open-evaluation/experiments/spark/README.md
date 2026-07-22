# Frozen Spark experiment code

This directory preserves the scripts used to produce the non-sensitive Spark PoC. Run them through `scripts/reproduce_spark.py`, which isolates temporary generated cohorts and avoids committing `.pkl`, model weights or DICOM files.

`00_public_index_qc.py` is optional and requires a local copy of the permitted public challenge archive/index. The input is intentionally not included.

`08_real_ct_pixel_evidence.py` is optional and requires the real-CT extras. It uses public package test fixtures and does not access challenge DICOM studies.
