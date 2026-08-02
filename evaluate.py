"""
evaluate.py — Compute SI-SNR / PESQ / STOI on a fixed eval set.

Expects:
    data/eval/noisy/eval_NNNN.wav
    data/eval/clean/eval_NNNN.wav

Usage:
    python evaluate.py
    python evaluate.py --noisy data/eval/noisy --clean data/eval/clean
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import librosa
import numpy as np
import torch

from src.models.voice_filter_model import (
    StudioEnhanceNet, reconstruct_audio,
    SR, N_FFT, HOP,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-7s | %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('eval')


# ── Metrics ───────────────────────────────────────────────────────────────────
def si_snr_np(est: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    n = min(len(est), len(target))
    e = est[:n]    - est[:n].mean()
    t = target[:n] - target[:n].mean()
    s = np.dot(e, t) / (np.dot(t, t) + eps) * t
    n_ = e - s
    return float(10.0 * np.log10((s ** 2).sum() / ((n_ ** 2).sum() + eps)))


def pesq_score(ref: np.ndarray, est: np.ndarray, sr: int) -> float | None:
    try:
        from pesq import pesq
        return float(pesq(sr, ref, est, 'wb'))
    except Exception:
        return None


def stoi_score(ref: np.ndarray, est: np.ndarray, sr: int) -> float | None:
    try:
        from pystoi import stoi
        return float(stoi(ref, est, sr, extended=False))
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
def enhance_with_model(model: StudioEnhanceNet, audio: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    if x.shape[-1] < N_FFT:
        x = torch.nn.functional.pad(x, (0, N_FFT - x.shape[-1]))
    win = torch.hann_window(N_FFT)
    spec = torch.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                      window=win, center=False, return_complex=True)
    with torch.no_grad():
        y_real, y_imag = model(spec.real, spec.imag)
        out = reconstruct_audio(y_real, y_imag, length=x.shape[-1])
    return out.squeeze(0).cpu().numpy().astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--noisy', default='data/eval/noisy')
    p.add_argument('--clean', default='data/eval/clean')
    p.add_argument('--ckpt',  default='checkpoints/best_model.pth')
    p.add_argument('--out',   default='eval_results/evaluation_report.csv')
    p.add_argument('--max',   type=int, default=200)
    args = p.parse_args()

    noisy_dir = Path(args.noisy)
    clean_dir = Path(args.clean)
    if not noisy_dir.exists() or not clean_dir.exists():
        log.error('eval directories missing — run create_mixtures.py first.')
        return

    model = StudioEnhanceNet()
    if Path(args.ckpt).exists():
        ckpt  = torch.load(args.ckpt, map_location='cpu')
        state = ckpt.get('model') or ckpt.get('state_dict') or ckpt
        model.load_state_dict(state, strict=False)
        log.info(f'Loaded checkpoint: {args.ckpt}')
    else:
        log.warning(f'No checkpoint at {args.ckpt} — evaluating with random weights.')
    model.eval()

    pairs = sorted(noisy_dir.glob('*.wav'))[:args.max]
    if not pairs:
        log.error(f'No WAVs found in {noisy_dir}')
        return

    rows = []
    for nf in pairs:
        cf = clean_dir / nf.name
        if not cf.exists():
            continue
        noisy, _ = librosa.load(str(nf), sr=SR, mono=True)
        clean, _ = librosa.load(str(cf), sr=SR, mono=True)
        n = min(len(noisy), len(clean))
        noisy, clean = noisy[:n], clean[:n]

        enhanced = enhance_with_model(model, noisy)[:n]

        baseline = si_snr_np(noisy, clean)
        after    = si_snr_np(enhanced, clean)
        pesq_val = pesq_score(clean, enhanced, SR)
        stoi_val = stoi_score(clean, enhanced, SR)

        rows.append({
            'file':     nf.name,
            'sisnr_in': round(baseline, 2),
            'sisnr_out':round(after,    2),
            'sisnr_d':  round(after - baseline, 2),
            'pesq':     round(pesq_val, 2) if pesq_val is not None else '',
            'stoi':     round(stoi_val, 3) if stoi_val is not None else '',
        })
        print(f'{nf.name:<20} SI-SNR {baseline:+.2f} -> {after:+.2f} dB  '
              f'(D {after-baseline:+.2f})')

    if rows:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        log.info(f'Wrote {out_path}')

        deltas = [r['sisnr_d'] for r in rows]
        log.info(f'Mean delta SI-SNR: {np.mean(deltas):+.2f} dB '
                 f'(N={len(deltas)})')


if __name__ == '__main__':
    main()
