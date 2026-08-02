"""
noise_reducer.py — Stage 2 of the inference pipeline.

Spectral-subtraction noise reduction via the `noisereduce` library.
"""
from __future__ import annotations

import os

import numpy as np


class NoiseReducer:
    def __init__(self,
                 prop_decrease: float = 0.85,
                 stationary:    bool  = False):
        self.enabled       = os.getenv('ENABLE_NOISEREDUCE', 'true').lower() == 'true'
        self.prop_decrease = prop_decrease
        self.stationary    = stationary

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if not self.enabled or audio.size == 0:
            return audio.astype(np.float32, copy=False)
        try:
            import noisereduce as nr
            return nr.reduce_noise(
                y             = audio.astype(np.float32),
                sr            = sr,
                stationary    = self.stationary,
                prop_decrease = self.prop_decrease,
            ).astype(np.float32)
        except Exception:
            return audio.astype(np.float32, copy=False)
