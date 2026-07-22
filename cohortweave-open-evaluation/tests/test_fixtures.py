import csv, gzip, json, struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_synthetic_csv():
    rows=list(csv.DictReader((ROOT/'fixtures/synthetic_clinical.csv').open()))
    assert len(rows)==64 and all(r['subject_id'].startswith('SYNTH_') for r in rows)
def test_nifti_header():
    with gzip.open(ROOT/'fixtures/synthetic_ct_phantom.nii.gz','rb') as f: h=f.read(352)
    assert struct.unpack_from('<i',h,0)[0]==348
    assert struct.unpack_from('<4h',h,40)[:4]==(3,32,32,16)
    assert h[344:348]==b'n+1\x00'
def test_manifest_hashes():
    import hashlib
    m=json.loads((ROOT/'fixtures/fixture_manifest.json').read_text())
    for name,meta in m.items(): assert hashlib.sha256((ROOT/'fixtures'/name).read_bytes()).hexdigest()==meta['sha256']
