"""
noise_gate.py — Aggressive post-DeepFilterNet noise gate.

DeepFilterNet reduces background noise but deliberately stops short of full
suppression (over-suppression destroys speech quality and creates "musical
noise" artifacts), so a residual noise floor typically remains between
words even after enhancement. This stage closes that gap explicitly:
energy-based VAD marks speech blocks, and everything else is pulled down
toward near-silence, scaled by `strength`. A hold time (keep the gate open
briefly after speech ends) and a slow release (fade down, not a hard cut)
keep trailing consonants and breath sounds from being chopped off audibly.

Exposes:
    apply_noise_gate(audio, sample_rate, strength, floor_db=-50.0) -> np.ndarray
"""
from __future__ import annotations

import numpy as np


def _block_reshape(audio: np.ndarray, block_len: int) -> np.ndarray:
    n_blocks = max(1, int(np.ceil(len(audio) / block_len)))
    padded = np.pad(audio, (0, n_blocks * block_len - len(audio)))
    return padded.reshape(n_blocks, block_len)


def apply_noise_gate(audio: np.ndarray, sr: int, strength: float,
                      floor_db: float = -50.0,
                      block_ms: float = 10.0,
                      hold_ms: float = 150.0,
                      attack_ms: float = 5.0,
                      release_ms: float = 200.0) -> np.ndarray:
    """Gate everything that isn't classified as speech down toward `floor_db`,
    scaled linearly by `strength` (0.0 = no-op, 1.0 = full gating to floor_db).
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    audio = np.asarray(audio, dtype=np.float32)
    if strength <= 0.0 or audio.size == 0:
        return audio.copy()

    try:
        block_len = max(1, int(sr * block_ms / 1000.0))
        blocks = _block_reshape(audio, block_len)
        n_blocks = blocks.shape[0]
        block_rms = np.sqrt(np.mean(blocks ** 2, axis=1) + 1e-12)
        block_db = 20.0 * np.log10(block_rms + 1e-12)

        noise_floor = np.percentile(block_db, 20)
        threshold = noise_floor + 9.0
        is_speech = block_db > threshold

        # Extend each speech block forward by `hold_blocks` so word tails and
        # breaths right after loud speech aren't gated the instant energy dips.
        hold_blocks = max(1, int(round(hold_ms / block_ms)))
        held = is_speech.copy()
        countdown = 0
        for i in range(n_blocks):
            if is_speech[i]:
                countdown = hold_blocks
            elif countdown > 0:
                held[i] = True
                countdown -= 1

        floor_lin = 10.0 ** (floor_db / 20.0)
        target_gain = np.where(held, 1.0, floor_lin)
        # strength=0 -> no gating at all; strength=1 -> full drop to floor_lin.
        target_gain = 1.0 - strength * (1.0 - target_gain)

        # Fast to open, slow to close, so the gate doesn't click or thump.
        block_rate = sr / block_len
        attack_coef = np.exp(-1.0 / (block_rate * attack_ms / 1000.0))
        release_coef = np.exp(-1.0 / (block_rate * release_ms / 1000.0))

        smoothed = np.empty(n_blocks, dtype=np.float64)
        level = target_gain[0]
        for i, g in enumerate(target_gain):
            coef = attack_coef if g > level else release_coef
            level = coef * level + (1.0 - coef) * g
            smoothed[i] = level

        gain_sample = np.repeat(smoothed, block_len)[: len(audio)]
        return (audio * gain_sample.astype(np.float32)).astype(np.float32)
    except Exception:
        return audio
