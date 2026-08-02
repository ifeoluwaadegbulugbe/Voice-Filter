"""
sb_encoder.py — Optional ECAPA-TDNN speaker embedding for identity verification.

Computes a 192-D embedding per audio clip, used by the /enroll and /verify
endpoints.  Lazy-loaded — silently disables itself if speechbrain isn't
installed or fails to download the weights.
"""
from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger('sb_encoder')


class ECAPAEncoder:
    def __init__(self):
        self._encoder = None
        self._available = False

        if os.getenv('ENABLE_SB_ENCODER', 'false').lower() != 'true':
            logger.info('ECAPA encoder disabled by env (ENABLE_SB_ENCODER=false).')
            return

        try:
            os.environ.setdefault('SB_DISABLE_K2', '1')
            os.environ.setdefault('K2_DISABLE',    '1')
            from speechbrain.inference import EncoderClassifier
            self._encoder = EncoderClassifier.from_hparams(
                source   = 'speechbrain/spkrec-ecapa-voxceleb',
                run_opts = {'device': 'cpu'},
            )
            self._available = True
            logger.info('ECAPA-TDNN encoder ready.')
        except Exception as e:
            logger.warning(f'ECAPA encoder unavailable: {e}')

    @property
    def is_available(self) -> bool:
        return self._available

    def embed(self, audio: np.ndarray, sr: int = 16_000) -> np.ndarray | None:
        if not self._available or self._encoder is None:
            return None
        try:
            import torch
            tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            emb    = self._encoder.encode_batch(tensor)
            return emb.squeeze().detach().cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.error(f'Embedding failed: {e}')
            return None

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = a.flatten().astype(np.float32)
        b = b.flatten().astype(np.float32)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
