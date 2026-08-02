"""
create_mixtures.py — Build a fixed evaluation set of (noisy, clean) pairs.

Run once after preprocessing to make eval reproducible across training runs.

Output:
    data/eval/noisy/eval_NNNN.wav
    data/eval/clean/eval_NNNN.wav
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-7s | %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('create_mixtures')

SR        = 16_000
FIXED_SEC = 3.0
FIXED_LEN = int(SR * FIXED_SEC)


def _load_fixed(path: Path, rng: random.Random) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    if len(audio) >= FIXED_LEN:
        start = rng.randint(0, len(audio) - FIXED_LEN)
        return audio[start: start + FIXED_LEN].astype(np.float32)
    return np.pad(audio, (0, FIXED_LEN - len(audio))).astype(np.float32)


def _peak_norm(a: np.ndarray, target: float = 0.9) -> np.ndarray:
    peak = float(np.max(np.abs(a)))
    return (a / (peak + 1e-8) * target) if peak > 1e-8 else a


def _mix_at_snr(sig: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    s_rms   = np.sqrt(np.mean(sig   ** 2) + 1e-10)
    n_rms   = np.sqrt(np.mean(noise ** 2) + 1e-10)
    target  = s_rms / (10.0 ** (snr_db / 20.0))
    return sig + noise * (target / n_rms)


def _synth_rir(rng: random.Random) -> np.ndarray:
    rt60      = rng.uniform(0.10, 0.80)
    duration  = min(rt60 * 1.5, 2.0)
    n_samples = int(SR * duration)
    t         = np.linspace(0, duration, n_samples)
    decay     = np.exp(-6.91 * t / rt60)
    rir       = (np.random.randn(n_samples) * decay).astype(np.float32)
    rir      /= (float(np.max(np.abs(rir))) + 1e-8)
    return rir


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clean-dir',  default='data/clean')
    p.add_argument('--noise-dir',  default='data/noise')
    p.add_argument('--output-dir', default='data/eval')
    p.add_argument('--num-examples', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    clean_files = sorted(Path(args.clean_dir).glob('*.wav'))
    noise_files = sorted(Path(args.noise_dir).glob('*.wav'))
    if not clean_files:
        log.error('No clean WAVs found. Run download_dataset.py first.')
        return

    out_noisy = Path(args.output_dir) / 'noisy'
    out_clean = Path(args.output_dir) / 'clean'
    out_noisy.mkdir(parents=True, exist_ok=True)
    out_clean.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i in range(args.num_examples):
        try:
            clean  = _load_fixed(rng.choice(clean_files), rng)
            clean  = _peak_norm(clean, 0.8)
            target = clean.copy()

            noisy = clean.copy()

            # Reverb
            if rng.random() < 0.7:
                rir   = _synth_rir(rng)
                noisy = fftconvolve(noisy, rir, mode='full')[:len(clean)].astype(np.float32)

            # Noise
            if rng.random() < 0.85:
                if noise_files:
                    noise = _load_fixed(rng.choice(noise_files), rng)
                else:
                    noise = np.random.randn(len(clean)).astype(np.float32)
                snr   = rng.uniform(-5.0, 15.0)
                noisy = _mix_at_snr(noisy, noise, snr)

            noisy  = _peak_norm(noisy,  0.9)
            target = _peak_norm(target, 0.9)

            tag = f'eval_{i:04d}'
            sf.write(str(out_noisy / f'{tag}.wav'), noisy,  SR, subtype='PCM_16')
            sf.write(str(out_clean / f'{tag}.wav'), target, SR, subtype='PCM_16')
            saved += 1
        except Exception as e:
            log.warning(f'  skipped {i}: {e}')

    log.info(f'Created {saved}/{args.num_examples} pairs at {args.output_dir}/')
    log.info('Next: python evaluate.py')


if __name__ == '__main__':
    main()
