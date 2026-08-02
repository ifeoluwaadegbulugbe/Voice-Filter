"""
verify_setup.py — Quick health check.
Verifies: dependencies installed, data dirs exist, checkpoint findable.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

PASS = '[ OK ]'
FAIL = '[FAIL]'
WARN = '[WARN]'

issues: list[str] = []


def check_module(name: str) -> None:
    try:
        importlib.import_module(name)
        print(f'  {PASS} {name}')
    except ImportError:
        print(f'  {FAIL} {name}  (missing — pip install {name})')
        issues.append(f'pip install {name}')


def check_path(p: str, kind: str = 'dir') -> None:
    P = Path(p)
    if kind == 'dir' and P.is_dir():
        n = len(list(P.glob('*')))
        print(f'  {PASS} {p}/  ({n} entries)')
    elif kind == 'file' and P.is_file():
        size_mb = P.stat().st_size / 1024 / 1024
        print(f'  {PASS} {p}  ({size_mb:.1f} MB)')
    else:
        print(f'  {WARN} {p} not found.')
        issues.append(f'create or populate {p}')


print('\n=== Voice Filter Project — Setup Verification ===\n')

print('Python packages:')
for m in ['torch', 'torchaudio', 'librosa', 'soundfile', 'numpy',
          'fastapi', 'uvicorn', 'noisereduce', 'tqdm']:
    check_module(m)

print('\nProject directories:')
for d in ['backend', 'ai_pipeline', 'preprocessing',
          'src/models', 'src/training', 'scripts',
          'data/clean', 'data/noise', 'checkpoints']:
    check_path(d)

print('\nKey files:')
check_path('checkpoints/best_model.pth', 'file')

print('\n=== Summary ===')
if issues:
    print(f'{len(issues)} issues to fix:')
    for i in issues:
        print(f'  - {i}')
    sys.exit(1)
else:
    print('All checks passed.')
