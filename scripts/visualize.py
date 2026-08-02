"""
visualize.py — Spectrogram + waveform before/after the AI enhancer.

Drop a noisy WAV in, get a side-by-side comparison PNG showing
exactly what the model is doing.

Usage:
    python scripts/visualize.py path/to/noisy.wav
    python scripts/visualize.py path/to/noisy.wav --clean path/to/clean.wav
    python scripts/visualize.py path/to/noisy.wav --out outputs/demo.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from ai_pipeline.sb_enhancer import StudioEnhancer


SR = 16_000


def _spec_db(audio: np.ndarray) -> np.ndarray:
    spec = np.abs(librosa.stft(audio, n_fft=512, hop_length=128))
    return librosa.amplitude_to_db(spec, ref=np.max)


def _si_snr(est: np.ndarray, target: np.ndarray) -> float:
    n = min(len(est), len(target))
    e = est[:n]    - est[:n].mean()
    t = target[:n] - target[:n].mean()
    s = np.dot(e, t) / (np.dot(t, t) + 1e-8) * t
    n_ = e - s
    return float(10.0 * np.log10((s ** 2).sum() / ((n_ ** 2).sum() + 1e-8)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input',          help='Noisy WAV to enhance & plot.')
    p.add_argument('--clean', default=None,
                   help='Optional clean reference WAV for ground-truth comparison.')
    p.add_argument('--ckpt',  default='checkpoints/best_model.pth')
    p.add_argument('--out',   default='outputs/visualize.png')
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    noisy, _ = librosa.load(args.input, sr=SR, mono=True)
    print(f'Loaded {args.input}  ({len(noisy)/SR:.1f} s)')

    enhancer = StudioEnhancer(ckpt_path=args.ckpt)
    enhanced = enhancer.enhance(noisy, sr=SR)
    enhanced = enhanced[:len(noisy)]

    clean = None
    if args.clean and Path(args.clean).exists():
        clean, _ = librosa.load(args.clean, sr=SR, mono=True)
        clean = clean[:len(noisy)]

    n_rows = 3 if clean is not None else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows + 1))

    rows = [('Noisy input',  noisy, 'tab:red'),
            ('Enhanced',     enhanced, 'tab:green')]
    if clean is not None:
        rows.append(('Clean reference', clean, 'tab:blue'))

    for i, (title, audio, color) in enumerate(rows):
        # Waveform
        ax_w = axes[i, 0]
        t = np.arange(len(audio)) / SR
        ax_w.plot(t, audio, color=color, linewidth=0.5)
        ax_w.set_title(f'{title} — waveform')
        ax_w.set_xlabel('time (s)')
        ax_w.set_ylabel('amplitude')
        ax_w.set_ylim(-1.0, 1.0)
        ax_w.grid(alpha=0.3)

        # Spectrogram
        ax_s = axes[i, 1]
        spec = _spec_db(audio)
        im = librosa.display.specshow(spec, sr=SR, hop_length=128,
                                      x_axis='time', y_axis='hz', ax=ax_s, cmap='magma')
        ax_s.set_title(f'{title} — spectrogram')
        plt.colorbar(im, ax=ax_s, format='%+2.0f dB', shrink=0.8)

    if clean is not None:
        baseline = _si_snr(noisy, clean)
        after    = _si_snr(enhanced, clean)
        fig.suptitle(
            f'SI-SNR  baseline {baseline:+.2f} dB  ->  enhanced {after:+.2f} dB  '
            f'(delta {after - baseline:+.2f} dB)',
            fontsize=14, y=1.005,
        )

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches='tight')
    print(f'Saved -> {args.out}')


if __name__ == '__main__':
    main()
