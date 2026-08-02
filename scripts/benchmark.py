"""
Benchmark — Hits /enhance and computes per-file SI-SNR delta.

Place pairs in test_audio/:
    test_audio/noisy_001.wav   (degraded input)
    test_audio/clean_001.wav   (clean reference)

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --api http://localhost:8000 --max 20
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import sys
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf

SR  = 16_000
API = 'http://localhost:8000'


def si_snr(est: np.ndarray, target: np.ndarray) -> float:
    n = min(len(est), len(target))
    e = est[:n]    - est[:n].mean()
    t = target[:n] - target[:n].mean()
    s = np.dot(e, t) / (np.dot(t, t) + 1e-8) * t
    n_ = e - s
    return float(10.0 * np.log10((s ** 2).sum() / ((n_ ** 2).sum() + 1e-8)))


def _wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SR, subtype='PCM_16', format='WAV')
    return buf.getvalue()


def call_enhance(audio: np.ndarray, api: str) -> np.ndarray | None:
    try:
        wav = _wav_bytes(audio)
        r = requests.post(f'{api}/enhance',
                          files={'audio': ('input.wav', wav, 'audio/wav')},
                          timeout=60)
        r.raise_for_status()
        raw = base64.b64decode(r.json()['enhanced_audio'])
        arr, _ = librosa.load(io.BytesIO(raw), sr=SR, mono=True)
        return arr.astype(np.float32)
    except Exception as e:
        print(f'  API error: {e}')
        return None


def run(api: str, test_dir: str, max_files: int):
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f'ERROR: {test_path.resolve()} does not exist.')
        sys.exit(1)
    noisy_files = sorted(test_path.glob('noisy_*.wav'))[:max_files]
    if not noisy_files:
        print(f'No noisy_*.wav files found in {test_dir}/')
        sys.exit(1)

    print(f'\n{"="*70}')
    print(f'BENCHMARK  api={api}   {len(noisy_files)} files')
    print(f'{"="*70}')
    print(f'{"file":<20} {"baseline":>10} {"enhanced":>10} {"delta":>8}')
    print(f'{"-"*70}')

    rows = []
    deltas = []
    for nf in noisy_files:
        idx = nf.stem.split('_', 1)[-1]
        cf  = test_path / f'clean_{idx}.wav'
        if not cf.exists():
            print(f'  [skip] no clean_{idx}.wav')
            continue
        noisy, _ = librosa.load(str(nf), sr=SR, mono=True)
        clean, _ = librosa.load(str(cf), sr=SR, mono=True)
        n = min(len(noisy), len(clean))
        noisy, clean = noisy[:n], clean[:n]

        baseline = si_snr(noisy, clean)
        enhanced = call_enhance(noisy, api)
        if enhanced is None:
            print(f'  {nf.name:<20} {baseline:>+10.2f}  [API FAILED]')
            continue
        enhanced = enhanced[:n]
        after = si_snr(enhanced, clean)
        delta = after - baseline
        deltas.append(delta)
        rows.append({'file': nf.name,
                     'baseline_db': round(baseline, 2),
                     'enhanced_db': round(after,    2),
                     'delta_db':    round(delta,    2)})
        print(f'  {nf.name:<20} {baseline:>+10.2f} {after:>+10.2f} {delta:>+8.2f}')

    print(f'{"="*70}')
    if deltas:
        print(f'Mean delta SI-SNR: {np.mean(deltas):+.2f} dB '
              f'+/- {np.std(deltas):.2f} dB  ({len(deltas)} files)')

    if rows:
        with open('benchmark_results.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print('\nResults -> benchmark_results.csv')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--api',      default=API)
    p.add_argument('--test-dir', default='test_audio')
    p.add_argument('--max',      type=int, default=50)
    args = p.parse_args()
    run(args.api, args.test_dir, args.max)
