#!/usr/bin/env python3
from __future__ import annotations
import subprocess, sys, zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(root/'scripts/check_public_release.py')],check=True)
out=root.parent/f'{root.name}-v0.1.0-spark.zip'
exclude={'.git','.pytest_cache','__pycache__','artifacts'}
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for p in root.rglob('*'):
  if not p.is_file() or any(x in p.parts for x in exclude): continue
  z.write(p,p.relative_to(root.parent))
print(out)
