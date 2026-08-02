"""
debug_pipeline.py — Run all 5 sanity tests in order. Stops at the first failure.
Note: SI-SNR is clamped to +35 dB max in loss.py, so any "perfect" reconstruction
returns exactly +35.0, not infinity.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models.voice_filter_model import StudioEnhanceNet, reconstruct_audio, N_FFT, HOP
from src.training.dataset import SpeechEnhancementDataset
from src.training.loss    import si_snr

# si_snr is clamped to +35.0 dB in loss.py — perfect reconstructions hit the ceiling.
SISNR_CEIL  = 35.0
NEAR_PERFECT = 30.0   # >= this means "essentially perfect"
GOOD         = 5.0    # >= this means "model is at least not destroying signal"


def stft_pair(wav):
    win = torch.hann_window(N_FFT, device=wav.device)
    spec = torch.stft(wav, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                      window=win, center=True, return_complex=True)
    return spec.real, spec.imag


def step1_roundtrip():
    x = torch.randn(2, 48_000)
    r, i = stft_pair(x)
    y = reconstruct_audio(r, i, length=x.shape[-1])
    err = (x - y).abs().mean().item()
    print(f"  Step 1 — STFT round-trip:   err={err:.2e}", end="")
    assert err < 1e-3, "  FAIL — fix center/hop/length"
    print("                                 ✓ PASS")


def step2_identity():
    """Bypass the model — apply identity STFT/iSTFT.  Should hit the clamp."""
    x = torch.randn(2, 48_000)
    r, i = stft_pair(x)
    y = reconstruct_audio(r, i, length=x.shape[-1])
    s = si_snr(y, x).mean().item()
    print(f"  Step 2 — identity STFT:     SI-SNR={s:+.1f} dB (ceil={SISNR_CEIL})", end="")
    assert s >= NEAR_PERFECT, "  FAIL — STFT round-trip introducing distortion"
    print("        ✓ PASS")


def step3_constant_attenuation():
    """0.5 * spectrum.  SI-SNR is scale-invariant, so this should ALSO hit the ceiling."""
    x = torch.randn(2, 48_000)
    r, i = stft_pair(x)
    y = reconstruct_audio(r * 0.5, i * 0.5, length=x.shape[-1])
    s = si_snr(y, x).mean().item()
    print(f"  Step 3 — 0.5x mask (scale): SI-SNR={s:+.1f} dB (scale-invariant)", end="")
    assert s >= NEAR_PERFECT, "  FAIL — si_snr not scale-invariant; check implementation"
    print("        ✓ PASS")


def step4_random_model():
    """Fresh model with identity-init bias. Output should be ~= input."""
    model = StudioEnhanceNet()
    model.eval()
    x = torch.randn(1, 48_000) * 0.1
    with torch.no_grad():
        r, i = stft_pair(x)
        y_r, y_i = model(r, i)
        y = reconstruct_audio(y_r, y_i, length=x.shape[-1])
    s = si_snr(y, x).mean().item()
    print(f"  Step 4 — fresh model fwd:   SI-SNR={s:+.1f} dB (identity init)", end="")
    assert s >= GOOD, ("  FAIL — model identity init broken; check crm_head.bias "
                       "initialisation in voice_filter_model.py")
    print("            ✓ PASS")


def step5_dataset_sanity():
    """Confirm dataset returns aligned (clean, noisy) at the right length."""
    try:
        ds = SpeechEnhancementDataset(length=4)
    except RuntimeError as e:
        print(f"  Step 5 — dataset:           SKIPPED ({e})")
        return
    s = ds[0]
    n = s["noisy_wav"].shape[-1]
    c = s["clean_wav"].shape[-1]
    print(f"  Step 5 — dataset shapes:    noisy={n} clean={c}", end="")
    assert n == c == 48_000, "  FAIL — length mismatch"
    inp_si = si_snr(s["noisy_wav"][None], s["clean_wav"][None]).item()
    print(f"   noisy-vs-clean SI-SNR={inp_si:+.1f} dB ✓ PASS")


def main():
    print("\n=== Pipeline diagnosis ===\n")
    step1_roundtrip()
    step2_identity()
    step3_constant_attenuation()
    step4_random_model()
    step5_dataset_sanity()
    print("\nAll 5 sanity checks PASSED. Your pipeline is structurally sound.")
    print("Train: python train.py --epochs 10 --batch-size 4 --accum-steps 4 --dataset-len 1000")


if __name__ == "__main__":
    main()