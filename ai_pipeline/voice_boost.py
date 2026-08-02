"""
voice_boost.py — Voice Loudness Enhancement stage.

Sits between the AI enhancer and the final limiter:
    SoxProcessor -> NoiseReducer -> StudioEnhancer -> VoiceBoost -> PostProcessor

DeepFilterNet has already suppressed background noise by this point, so this
stage focuses purely on making the remaining speech sound like a mastered
podcast, in three passes:

  1. Leveler   — a slow (per ~200ms block), VAD-gated AGC that evens out
                 loudness *locally* across sentences/phrases: quiet passages
                 get pulled up toward a target level, already-loud passages
                 get little/no gain. This runs first so no later stage has to
                 reconcile "the loud part earlier in the clip" with "the
                 quiet part now" — each block is judged on its own level.
  2. Compressor — a fast (10ms/100ms) soft-knee downward compressor that
                 catches transient peaks the leveler's slower response
                 can't, with a capped automatic makeup gain.
  3. Loudness match — a small corrective gain (bounded, so it can't undo
                 what the leveler already decided) to land the overall
                 integrated loudness near the -16/-18 LUFS target.

A hard, peak-exact limiter (scale-to-peak, the same pattern already used by
SoxProcessor/PostProcessor elsewhere in this codebase) guarantees the output
never exceeds the ceiling — unlike a bare tanh curve, which asymptotically
overshoots for hot input and does not actually bound the output.

Exposes:
    boost_voice(audio, sample_rate, strength=0.7) -> np.ndarray   (one-shot)
    VoiceBoost(sample_rate, strength)                             (stateful,
        for streaming use where the leveler/compressor state and loudness
        estimate should persist across chunks instead of resetting)
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("voice_boost")

try:
    import pyloudnorm as pyln
    _HAS_PYLOUDNORM = True
except Exception:
    _HAS_PYLOUDNORM = False

QUIET_LUFS = -70.0  # floor used for near-silent audio to avoid -inf


def _measure_lufs(audio: np.ndarray, sr: int) -> float:
    """Integrated loudness in LUFS. pyloudnorm implements ITU-R BS.1770 gating,
    which already ignores silence/near-silence blocks, so no manual VAD gating
    is needed here."""
    if audio.size == 0:
        return QUIET_LUFS
    if _HAS_PYLOUDNORM:
        try:
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio.astype(np.float64))
            if np.isfinite(loudness):
                return float(loudness)
        except Exception as e:
            logger.debug(f"pyloudnorm failed ({e}); falling back to RMS")
    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    if rms <= 1e-6:
        return QUIET_LUFS
    # Rough RMS(dBFS) -> LUFS approximation for speech-like signals.
    return 20.0 * np.log10(rms) - 0.7


def _block_reshape(audio: np.ndarray, block_len: int) -> np.ndarray:
    n_blocks = max(1, int(np.ceil(len(audio) / block_len)))
    padded = np.pad(audio, (0, n_blocks * block_len - len(audio)))
    return padded.reshape(n_blocks, block_len)


def _energy_vad(audio: np.ndarray, sr: int, frame_ms: float = 20.0) -> np.ndarray:
    """Lightweight energy-based VAD (no external model needed). Returns a
    per-sample soft mask in [0, 1] marking where speech is likely present,
    so gain can be steered away from silence/residual noise."""
    frame_len = max(1, int(sr * frame_ms / 1000))
    frames = _block_reshape(audio, frame_len)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    frame_db = 20.0 * np.log10(frame_rms + 1e-12)

    noise_floor = np.percentile(frame_db, 20)
    threshold = noise_floor + 10.0  # 10 dB above the estimated noise floor

    mask = (frame_db > threshold).astype(np.float32)
    if len(mask) >= 3:
        mask = np.convolve(mask, np.ones(3) / 3.0, mode="same")  # soften on/off edges
    mask = np.clip(mask, 0.0, 1.0)

    return np.repeat(mask, frame_len)[: len(audio)].astype(np.float32)


def _hard_limit(audio: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    """Peak-exact limiter: scales the whole buffer down so its peak sits at
    the ceiling, guaranteeing no sample ever exceeds it. (A bare tanh curve
    looks like a limiter but doesn't bound its output for hot input — this
    does, the same way SoxProcessor/PostProcessor normalize by measured
    peak elsewhere in this codebase.)"""
    ceiling_lin = 10.0 ** (ceiling_db / 20.0)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= ceiling_lin or peak < 1e-9:
        return audio.astype(np.float32)
    return (audio * (ceiling_lin / peak)).astype(np.float32)


class VoiceBoost:
    """Stateful voice-loudness booster.

    Carries the leveler/compressor envelopes and a smoothed loudness estimate
    across calls, so `process()` can be called repeatedly on successive
    streaming chunks without clicks/pumping at chunk boundaries. For one-shot
    file processing, use the `boost_voice()` module function instead.
    """

    def __init__(self, sample_rate: int, strength: float = 0.7):
        self.sr = sample_rate
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self._level_gain_db = 0.0   # leveler's smoothed gain, carried across calls
        self._env_level = 0.0       # compressor envelope, carried across calls
        self._lufs_ema: float | None = None

    def process(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0 or self.strength <= 0.0:
            return audio.copy()

        try:
            vad_mask = _energy_vad(audio, self.sr)

            # 1. Leveler — local (per ~200ms block) loudness evening, VAD-gated.
            leveled = self._level(audio, vad_mask)

            # 2. Compressor — fast peak control + small capped makeup gain.
            ratio = 1.0 + self.strength * 2.0  # 1:1 (off) .. 3:1 (full strength)
            compressed = self._compress(leveled, ratio=ratio, threshold_db=-20.0,
                                         knee_db=6.0, attack_ms=10.0, release_ms=100.0)

            # 3. Loudness match — small, bounded corrective gain toward target LUFS.
            measured_lufs = _measure_lufs(compressed, self.sr)
            self._lufs_ema = (
                measured_lufs if self._lufs_ema is None
                else 0.7 * self._lufs_ema + 0.3 * measured_lufs
            )
            target_lufs = -18.0 + self.strength * 2.0  # -18 (gentle) .. -16 (podcast)
            raw_gain_db = target_lufs - self._lufs_ema

            if self._lufs_ema <= QUIET_LUFS + 1.0:
                trim_db = 0.0  # near-silent chunk: don't amplify noise
            elif raw_gain_db >= 0:
                trim_db = min(raw_gain_db, 6.0) * self.strength
            else:
                trim_db = max(raw_gain_db, -2.0) * self.strength * 0.3

            trim_lin = 10.0 ** (trim_db / 20.0)
            gain_curve = 1.0 + (trim_lin - 1.0) * vad_mask  # gate by speech presence
            boosted = compressed * gain_curve

            return _hard_limit(boosted, ceiling_db=-1.0)
        except Exception as e:
            logger.warning(f"voice_boost failed ({e}); returning original audio")
            return audio

    def _level(self, audio: np.ndarray, vad_mask: np.ndarray,
               block_ms: float = 200.0, target_db: float = -20.0) -> np.ndarray:
        """Slow AGC: judges each ~200ms block on its own level (not the whole
        clip), so a loud section elsewhere in the recording can't suppress
        the boost a quiet section needs. Gain is scaled by how much of the
        block is actually speech (VAD), so silence/noise-only blocks are
        left alone."""
        block_len = max(1, int(self.sr * block_ms / 1000.0))
        blocks = _block_reshape(audio, block_len)
        vad_blocks = _block_reshape(vad_mask, block_len)

        block_rms = np.sqrt(np.mean(blocks ** 2, axis=1) + 1e-12)
        block_vad_frac = np.mean(vad_blocks, axis=1)
        block_db = 20.0 * np.log10(block_rms + 1e-12)

        raw_gain_db = target_db - block_db
        gain_db = np.where(
            raw_gain_db >= 0,
            np.minimum(raw_gain_db, 18.0) * self.strength,       # quiet -> boost
            np.maximum(raw_gain_db, -6.0) * self.strength * 0.4,  # loud -> little/no gain
        )
        gain_db = gain_db * block_vad_frac  # don't adjust gain where there's no speech

        # Smooth block-to-block so gain rides phrases, not individual samples
        # (asymmetric: react a bit faster when raising a quiet block than
        # when backing off a loud one, like a real leveler).
        smoothed = np.empty_like(gain_db)
        level = self._level_gain_db
        for i, g in enumerate(gain_db):
            alpha = 0.5 if g > level else 0.25
            level = alpha * g + (1.0 - alpha) * level
            smoothed[i] = level
        self._level_gain_db = float(level)  # persist for the next chunk

        gain_lin_block = 10.0 ** (smoothed / 20.0)
        gain_sample = np.repeat(gain_lin_block, block_len)[: len(audio)]
        return (audio * gain_sample.astype(np.float32)).astype(np.float32)

    def _compress(self, audio: np.ndarray, ratio: float, threshold_db: float,
                   knee_db: float, attack_ms: float, release_ms: float,
                   block_ms: float = 5.0) -> np.ndarray:
        """Feed-forward soft-knee downward compressor with a capped automatic
        makeup gain, processed in small blocks (not per-sample) so it stays
        fast on consumer CPUs while still tracking envelope with attack/release."""
        if ratio <= 1.0:
            return audio

        block_len = max(1, int(self.sr * block_ms / 1000.0))
        blocks = _block_reshape(audio, block_len)
        block_level = np.sqrt(np.mean(blocks ** 2, axis=1) + 1e-12)

        block_rate = self.sr / block_len
        attack_coef = np.exp(-1.0 / (block_rate * attack_ms / 1000.0))
        release_coef = np.exp(-1.0 / (block_rate * release_ms / 1000.0))

        env = np.empty(len(block_level), dtype=np.float64)
        level = self._env_level or block_level[0]
        for i, v in enumerate(block_level):
            coef = attack_coef if v > level else release_coef
            level = coef * level + (1.0 - coef) * v
            env[i] = level
        self._env_level = float(level)  # persist for the next chunk

        env_db = 20.0 * np.log10(env + 1e-12)
        over_db = env_db - threshold_db
        half_knee = knee_db / 2.0

        gain_reduction_db = np.zeros(len(block_level))
        above = over_db >= half_knee
        within = (over_db > -half_knee) & (~above)

        gain_reduction_db[above] = over_db[above] * (1.0 / ratio - 1.0)
        if np.any(within):
            knee_x = over_db[within] + half_knee
            gain_reduction_db[within] = (1.0 / ratio - 1.0) * (knee_x ** 2) / (2.0 * knee_db)

        reduced = gain_reduction_db < 0
        makeup_db = -0.5 * np.mean(gain_reduction_db[reduced]) if np.any(reduced) else 0.0
        makeup_db = min(makeup_db, 6.0)  # capped so it can't dominate the leveler's decisions
        gain_reduction_db = gain_reduction_db + makeup_db

        gain_lin_block = 10.0 ** (gain_reduction_db / 20.0)
        gain_sample = np.repeat(gain_lin_block, block_len)[: len(audio)]

        smooth_len = max(1, block_len // 2)
        if smooth_len > 1 and len(gain_sample) > smooth_len:
            gain_sample = np.convolve(gain_sample, np.ones(smooth_len) / smooth_len, mode="same")

        return (audio * gain_sample.astype(np.float32)).astype(np.float32)


def boost_voice(audio: np.ndarray, sample_rate: int, strength: float = 0.7) -> np.ndarray:
    """One-shot voice loudness boost for a full audio array.

    For streaming/chunked use (persistent leveler/compressor state and
    loudness estimate across calls), keep a `VoiceBoost` instance per
    connection and call `.process()` on each chunk instead.
    """
    return VoiceBoost(sample_rate, strength).process(audio)
