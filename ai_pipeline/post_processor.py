"""
post_processor.py — Stage 4 of the inference pipeline.

Light cosmetic polish after the AI enhancer:
  • DC blocker
  • High-pass at 80 Hz to kill rumble
  • Soft compressor / brick-wall limiter at -1 dBFS
  • Peak normalisation to -3 dBFS

Operates on numpy float32 arrays.  No torch dependency.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


class PostProcessor:
    def __init__(self,
                 sr:           int   = 16_000,
                 hp_hz:        float = 80.0,
                 limit_ceil:   float = 0.89,    # ~-1 dBFS
                 target_peak:  float = 0.71):   # ~-3 dBFS
        self.sr         = sr
        self.hp_sos     = butter(4, hp_hz, btype='highpass', fs=sr, output='sos')
        self.limit      = limit_ceil
        self.target     = target_peak

    # ── DC blocker (simple HP via running mean subtraction) ─────────────────
    @staticmethod
    def _dc_block(x: np.ndarray) -> np.ndarray:
        return (x - np.mean(x)).astype(np.float32)

    # ── Brick-wall limiter using tanh ───────────────────────────────────────
    def _limit(self, x: np.ndarray) -> np.ndarray:
        if self.limit <= 0:
            return x
        scale = 1.0 / np.tanh(self.limit)
        return np.tanh(x * self.limit) * scale

    # ── Public API ──────────────────────────────────────────────────────────
    def process(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio.astype(np.float32, copy=False)

        x = audio.astype(np.float32, copy=False)
        x = self._dc_block(x)
        x = sosfilt(self.hp_sos, x).astype(np.float32)
        x = self._limit(x)

        peak = float(np.max(np.abs(x))) or 1.0
        if peak > 1e-8:
            x = (x / peak * self.target).astype(np.float32)
        return x
