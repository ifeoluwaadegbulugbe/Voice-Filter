"""
pyannote_embed.py — Optional Voice Activity Detection (VAD).

Used by the server only when ENABLE_PYANNOTE=true and a valid HF_TOKEN is
present.  Skips initialisation silently otherwise; downstream code checks
.has_vad before calling .speech_mask.
"""
from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger('pyannote_embed')


class PyannoteVAD:
    def __init__(self):
        self._vad = None
        self._available = False

        if os.getenv('ENABLE_PYANNOTE', 'false').lower() != 'true':
            logger.info('pyannote VAD disabled by env (ENABLE_PYANNOTE=false).')
            return

        token = os.getenv('HF_TOKEN', '').strip()
        if not token.startswith('hf_'):
            logger.warning('HF_TOKEN missing or invalid; VAD disabled.')
            return

        try:
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained(
                'pyannote/voice-activity-detection',
                token=token,
            )
            self._vad = pipeline                 # IMPORTANT: keep reference
            self._available = True
            logger.info('pyannote VAD ready.')
        except Exception as e:
            logger.error(f'Failed to load pyannote VAD: {e}')

    @property
    def has_vad(self) -> bool:
        return self._available

    # ── Returns a binary speech mask aligned with the input samples ─────────
    def speech_mask(self, audio: np.ndarray, sr: int = 16_000) -> np.ndarray:
        if not self._available or self._vad is None:
            return np.ones_like(audio, dtype=np.float32)
        try:
            import torch
            tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            vad_out = self._vad({'waveform': tensor, 'sample_rate': sr})
            mask = np.zeros(len(audio), dtype=np.float32)
            for seg in vad_out.itersegments():
                start = max(0, int(seg.start * sr))
                end   = min(len(audio), int(seg.end * sr))
                mask[start:end] = 1.0
            return mask
        except Exception as e:
            logger.warning(f'VAD failed at runtime; passing audio through: {e}')
            return np.ones_like(audio, dtype=np.float32)
