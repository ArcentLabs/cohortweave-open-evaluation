from __future__ import annotations
import re
from pathlib import Path

BLOCKED_EXTENSIONS={'.dcm','.dicom','.pkl','.pickle','.pt','.pth','.ckpt','.onnx','.h5','.hdf5','.npz','.npy'}
BLOCKED_NAME_TERMS={'cv','curriculum_vitae','letter_of_intent','loi','partner_letter','application_form','concept_note'}
SECRET_PATTERNS={
 'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
 'github_token':re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}'),
 'aws_key':re.compile(r'AKIA[0-9A-Z]{16}'),
 'generic_api_key':re.compile(r'(?i)(api[_-]?key|client[_-]?secret|access[_-]?token)\s*[:=]\s*["\']?[A-Za-z0-9_\-]{16,}'),
}
TEXT_EXTENSIONS={'.md','.txt','.py','.toml','.yml','.yaml','.json','.csv','.cff','.ini','.cfg'}


def audit(root: Path) -> list[str]:
    root=Path(root); issues=[]
    for p in root.rglob('*'):
        if not p.is_file() or '.git' in p.parts: continue
        rel=p.relative_to(root)
        low=p.name.lower()
        if p.suffix.lower() in BLOCKED_EXTENSIONS:
            issues.append(f'blocked binary/model/medical extension: {rel}')
        if any(term in low for term in BLOCKED_NAME_TERMS):
            issues.append(f'blocked application/CV/partner-letter filename: {rel}')
        if p.stat().st_size > 10_000_000:
            issues.append(f'file exceeds public-release size threshold: {rel}')
        if p.suffix.lower() in TEXT_EXTENSIONS or p.name in {'Dockerfile','Makefile','LICENSE'}:
            try: text=p.read_text(encoding='utf-8')
            except UnicodeDecodeError: continue
            for name,pat in SECRET_PATTERNS.items():
                if pat.search(text): issues.append(f'possible {name} in {rel}')
    return sorted(set(issues))
