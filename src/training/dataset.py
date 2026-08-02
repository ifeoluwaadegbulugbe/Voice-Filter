"""SpeechEnhancementDataset — train (dynamic mixing) + eval_mode (paired)."""
from __future__ import annotations

import random
import warnings
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.signal import butter, fftconvolve, sosfilt
from torch.utils.data import Dataset


SR, FIXED_SEC = 16_000, 3.0
FIXED_LEN = int(SR * FIXED_SEC)


def _load(path, rng=None):
    """
    Load a wav, de-mean, crop to FIXED_LEN.
    Returns None if the file is corrupt — callers must handle None.
    rng=None gives deterministic crop (eval mode).
    """
    try:
        a, _ = librosa.load(str(path), sr=SR, mono=True)
    except Exception as e:
        warnings.warn(f"[dataset] Skipping corrupt file {Path(path).name}: {e}")
        return None
    a = a.astype(np.float32) - float(np.mean(a))
    if len(a) >= FIXED_LEN:
        s = (rng.randint(0, len(a) - FIXED_LEN) if rng is not None else 0)
        return a[s:s + FIXED_LEN]
    return np.pad(a, (0, FIXED_LEN - len(a)))


def _safe_choice(file_list, rng):
    """
    Pick a random file from file_list, retry until _load succeeds.
    Raises RuntimeError if no readable file is found after 20 attempts.
    """
    attempts = 0
    while attempts < 20:
        path = rng.choice(file_list)
        audio = _load(path, rng)
        if audio is not None:
            return audio
        attempts += 1
    raise RuntimeError(
        f"[dataset] Could not load any readable file after 20 attempts. "
        f"Check your data folder for corrupt .wav files."
    )


def _peak_norm(a, t=0.9):
    p = float(np.max(np.abs(a)))
    return (a / (p + 1e-8) * t) if p > 1e-8 else a


def _mix_at_snr(s, n, snr_db):
    s_rms = np.sqrt(np.mean(s ** 2) + 1e-10)
    n_rms = np.sqrt(np.mean(n ** 2) + 1e-10)
    return s + n * ((s_rms / (10.0 ** (snr_db / 20.0))) / n_rms)


def _telephone(a):
    sos = butter(6, [300.0, 3400.0], btype='bandpass', fs=SR, output='sos')
    return sosfilt(sos, a).astype(np.float32)


def _mulaw(a, mu=255):
    a = np.clip(a, -1.0, 1.0)
    e = np.sign(a) * np.log1p(mu * np.abs(a)) / np.log1p(mu)
    q = np.round(e * 127.0) / 127.0
    return (np.sign(q) * (1.0 / mu) * ((1.0 + mu) ** np.abs(q) - 1.0)).astype(np.float32)


def _synth_rir(rng):
    """
    Synthesise a small RIR whose direct path is the very first sample.
    Without this, fftconvolve introduces a random time-shift between clean
    and noisy, which causes SI-SNR to collapse — the model cannot match
    a moving target.
    """
    rt60 = rng.uniform(0.10, 0.60)
    dur  = min(rt60 * 1.5, 1.0)
    n    = int(SR * dur)
    rir  = np.zeros(n, dtype=np.float32)
    rir[0] = 1.0                                # direct path at sample 0
    gap = int(SR * 0.005)                       # 5 ms before first reflection
    if n > gap + 1:
        decay = np.exp(-6.91 * np.linspace(0, dur, n - gap) / rt60)
        rir[gap:] = (np.random.randn(n - gap).astype(np.float32) * decay * 0.3)
    return rir                                  # do NOT peak-normalise


def _random_eq(audio, rng):
    cutoff = float(rng.uniform(2_000.0, 7_000.0))
    btype  = 'low' if rng.random() < 0.5 else 'high'
    sos = butter(2, cutoff, btype=btype, fs=SR, output='sos')
    return sosfilt(sos, audio).astype(np.float32)


def _spec_mask(audio, rng):
    spec = librosa.stft(audio, n_fft=512, hop_length=128, center=True)
    F_, T_ = spec.shape
    if rng.random() < 0.5 and F_ > 6:
        f0 = rng.randint(0, F_ - 6)
        spec[f0:f0 + 6] = 0
    if rng.random() < 0.5 and T_ > 25:
        t0 = rng.randint(0, T_ - 25)
        spec[:, t0:t0 + 25] = 0
    out = librosa.istft(spec, hop_length=128, length=len(audio), center=True)
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────

class SpeechEnhancementDataset(Dataset):
    def __init__(
        self,
        clean_dir='data/clean',
        noise_dir='data/noise',
        rir_dir='data/rir',
        noisy_dir='data/noisy',
        length=2_000,
        seed=42,
        eval_mode=False,
    ):
        self.rng       = random.Random(seed)
        self.eval_mode = eval_mode
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)

        if eval_mode:
            self.eval_pairs = []
            if self.clean_dir.exists():
                for cf in sorted(self.clean_dir.glob('*.wav')):
                    nf = self.noisy_dir / cf.name
                    if nf.exists():
                        self.eval_pairs.append((cf, nf))
            self.length = max(1, len(self.eval_pairs))
            if not self.eval_pairs:
                print(f'[dataset] WARN: no paired files in {clean_dir} <-> {noisy_dir}')
        else:
            self.length      = length
            self.clean_files = sorted(self.clean_dir.glob('*.wav'))
            self.noise_files = sorted(Path(noise_dir).glob('*.wav'))
            self.rir_files   = sorted(Path(rir_dir).glob('*.wav'))
            if not self.clean_files:
                raise RuntimeError(f'No clean WAVs in {clean_dir}/')
            print(f'[dataset] clean={len(self.clean_files)}  '
                  f'noise={len(self.noise_files)}  '
                  f'rir={len(self.rir_files)}')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # ── Eval mode: return fixed paired files ─────────────────────────────
        if self.eval_mode:
            if not self.eval_pairs:
                z = np.zeros(FIXED_LEN, dtype=np.float32)
                return {
                    'clean_wav': torch.from_numpy(z),
                    'noisy_wav': torch.from_numpy(z),
                }
            cf, nf = self.eval_pairs[idx % len(self.eval_pairs)]
            clean_audio = _load(cf)
            noisy_audio = _load(nf)
            # fallback to zeros if eval file is also corrupt
            if clean_audio is None:
                clean_audio = np.zeros(FIXED_LEN, dtype=np.float32)
            if noisy_audio is None:
                noisy_audio = np.zeros(FIXED_LEN, dtype=np.float32)
            return {
                'clean_wav': torch.from_numpy(clean_audio),
                'noisy_wav': torch.from_numpy(noisy_audio),
            }

        # ── Train mode: dynamic mixing ────────────────────────────────────────
        clean = _peak_norm(_safe_choice(self.clean_files, self.rng), 0.8)
        noisy = clean.copy()

        # 1. Reverb (alignment-preserving: direct path at sample 0)
        if self.rng.random() < 0.5:
            if self.rir_files:
                rir_audio = _load(self.rng.choice(self.rir_files), self.rng)
                rir = rir_audio if rir_audio is not None else _synth_rir(self.rng)
            else:
                rir = _synth_rir(self.rng)
            noisy = fftconvolve(noisy, rir, mode='full')[:len(clean)].astype(np.float32)

        # 2. Primary noise
        if self.rng.random() < 0.85:
            if self.noise_files:
                n1 = _load(self.rng.choice(self.noise_files), self.rng)
                if n1 is None:
                    n1 = np.random.randn(len(clean)).astype(np.float32)
            else:
                n1 = np.random.randn(len(clean)).astype(np.float32)
            noisy = _mix_at_snr(noisy, n1, self.rng.uniform(-5.0, 20.0))

        # 3. Secondary simultaneous noise
        if self.rng.random() < 0.20 and len(self.noise_files) >= 2:
            n2 = _load(self.rng.choice(self.noise_files), self.rng)
            if n2 is not None:
                noisy = _mix_at_snr(noisy, n2, self.rng.uniform(0.0, 15.0))

        # 4. Augmentations
        if self.rng.random() < 0.15:
            noisy = _random_eq(noisy, self.rng)
        if self.rng.random() < 0.30:
            noisy = _telephone(noisy)
        if self.rng.random() < 0.30:
            noisy = _mulaw(noisy)
        if self.rng.random() < 0.10:
            try:
                noisy = _spec_mask(noisy, self.rng)
            except Exception:
                pass

        return {
            'clean_wav': torch.from_numpy(_peak_norm(clean, 0.9)),
            'noisy_wav': torch.from_numpy(_peak_norm(noisy, 0.9)),
        }