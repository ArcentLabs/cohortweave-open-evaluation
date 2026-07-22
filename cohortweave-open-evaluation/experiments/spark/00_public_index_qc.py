"""Public-index ingestion/QC for the AI-BOOST Challenge 3 supporting dataset.

This script validates metadata only. It does not access protected clinical JSON
or DICOM pixels and therefore is not model-performance or clinical-validation
evidence.

Usage:
  python 00_public_index_qc.py --archive /path/to/14046442.zip
  python 00_public_index_qc.py --index /path/to/index.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
HEX24 = re.compile(r"^[0-9a-f]{24}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DIGITS = re.compile(r"^[0-9]+$")


def load_index(index_path: Path | None, archive_path: Path | None) -> tuple[list[dict[str, Any]], bytes, str]:
    if archive_path:
        with zipfile.ZipFile(archive_path) as zf:
            raw = zf.read("index.json")
        source = archive_path.name
    elif index_path:
        raw = index_path.read_bytes()
        source = index_path.name
    else:
        raise ValueError("Provide --archive or --index")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise TypeError("index.json must contain a list")
    return data, raw, source


def validate(data: list[dict[str, Any]], raw: bytes, source: str) -> dict[str, Any]:
    required = {"studyId", "studyName", "subjectName", "url", "series"}
    schema_errors: list[str] = []
    study_ids: list[str] = []
    subject_ids: list[str] = []
    series_ids: list[str] = []
    tagged_per_study: list[int] = []
    series_counts: list[int] = []
    study_names: list[str] = []

    for i, study in enumerate(data):
        missing = required - set(study)
        if missing:
            schema_errors.append(f"study[{i}] missing {sorted(missing)}")
            continue
        study_ids.append(str(study["studyId"]))
        subject_ids.append(str(study["subjectName"]))
        study_names.append(str(study["studyName"]))
        series = study["series"]
        if not isinstance(series, list):
            schema_errors.append(f"study[{i}].series is not a list")
            continue
        series_counts.append(len(series))
        tagged = 0
        for j, item in enumerate(series):
            if not isinstance(item, dict) or "folderName" not in item or "tags" not in item:
                schema_errors.append(f"study[{i}].series[{j}] malformed")
                continue
            series_ids.append(str(item["folderName"]))
            if "Train/Val" in item.get("tags", []):
                tagged += 1
        tagged_per_study.append(tagged)

    n_series = len(series_ids)
    result = {
        "scope": {
            "evidence_type": "public metadata ingestion/QC only",
            "not_included": ["protected clinical JSON", "DICOM pixels", "model training", "clinical validation"],
            "source": source,
            "index_sha256": hashlib.sha256(raw).hexdigest(),
        },
        "counts": {
            "studies": len(data),
            "unique_study_ids": len(set(study_ids)),
            "unique_subject_hashes": len(set(subject_ids)),
            "series": n_series,
            "unique_series_folder_ids": len(set(series_ids)),
            "distinct_study_name_strings": len(set(study_names)),
            "series_per_study_min": min(series_counts) if series_counts else None,
            "series_per_study_median": sorted(series_counts)[len(series_counts)//2] if series_counts else None,
            "series_per_study_mean": round(sum(series_counts) / len(series_counts), 3) if series_counts else None,
            "series_per_study_max": max(series_counts) if series_counts else None,
        },
        "validation": {
            "schema_errors": len(schema_errors),
            "duplicate_study_ids": len(study_ids) - len(set(study_ids)),
            "duplicate_subject_hashes": len(subject_ids) - len(set(subject_ids)),
            "duplicate_series_folder_ids": len(series_ids) - len(set(series_ids)),
            "valid_study_id_format_pct": round(100 * sum(bool(HEX24.match(x)) for x in study_ids) / max(1, len(study_ids)), 2),
            "valid_subject_hash_format_pct": round(100 * sum(bool(HEX64.match(x)) for x in subject_ids) / max(1, len(subject_ids)), 2),
            "valid_series_folder_format_pct": round(100 * sum(bool(DIGITS.match(x)) for x in series_ids) / max(1, len(series_ids)), 2),
            "studies_with_exactly_one_train_val_tag": sum(x == 1 for x in tagged_per_study),
            "train_val_tag_count_distribution": dict(sorted(Counter(tagged_per_study).items())),
        },
        "interpretation": (
            "The public index demonstrates ingestion/QC readiness at full metadata scale. "
            "Distinct study-name strings are not treated as validated acquisition protocols, "
            "and the indexed series count is not assumed to be the final Challenge 3 training workload."
        ),
    }
    if schema_errors:
        result["validation"]["schema_error_examples"] = schema_errors[:10]
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--archive", type=Path)
    group.add_argument("--index", type=Path)
    ap.add_argument("--output", type=Path, default=BASE / "index_qc_results.json")
    args = ap.parse_args()
    data, raw, source = load_index(args.index, args.archive)
    result = validate(data, raw, source)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
