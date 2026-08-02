"""
sox_processor.py — Stage 1 of the inference pipeline.

Pure-Python equivalent of `sox`:
  • Resample to 16 kHz mono
  • DC removal
  • Normalize peak to a target level
"""
from __future__ import annotations

import os

import librosa
import numpy as np


class SoxProcessor:
    def __init__(self, target_sr: int = 16_000, target_peak: float = 0.85):
        self.enabled    = os.getenv('ENABLE_SOX', 'true').lower() == 'true'
        self.target_sr  = target_sr
        self.target_peak = target_peak

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        if not self.enabled or audio.size == 0:
            return audio.astype(np.float32, copy=False), sr

        x = audio.astype(np.float32, copy=False)
        if x.ndim > 1:
            x = x.mean(axis=-1)
        if sr != self.target_sr:
            x = librosa.resample(x, orig_sr=sr, target_sr=self.target_sr)
        x = x - float(np.mean(x))
        peak = float(np.max(np.abs(x)))
        if peak > 1e-8:
            x = x / peak * self.target_peak
        return x.astype(np.float32), self.target_sr
