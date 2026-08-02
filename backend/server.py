"""
backend/server.py — FastAPI server (Improved Stable Version)

Endpoints:
  GET  /status                Health + model info
  POST /filter               Returns enhanced WAV directly (audio/wav)
  POST /enhance/wav          Alias
  WS   /ws/stream            Real-time PCM streaming
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor

import librosa
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, File, Query, Response, UploadFile, HTTPException, WebSocket

from ai_pipeline.sb_enhancer import StudioEnhancer
from ai_pipeline.post_processor import PostProcessor
from ai_pipeline.voice_boost import VoiceBoost
from ai_pipeline.noise_gate import apply_noise_gate
from preprocessing.sox_processor import SoxProcessor
from preprocessing.noise_reducer import NoiseReducer
from backend.ws_handler import handle_stream

load_dotenv()

# ── Logging ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("server")

SR = int(os.getenv("TARGET_SR", "16000"))

# ── Voice Boost defaults (overridable per-request) ──────
DEFAULT_VOICE_BOOST      = os.getenv("ENABLE_VOICE_BOOST", "true").lower() == "true"
DEFAULT_BOOST_STRENGTH   = float(os.getenv("VOICE_BOOST_STRENGTH", "0.7"))

# ── Noise reduction strength default (overridable per-request) ──
# 0.0 = pass the pre-noise-reduction signal through untouched,
# 1.0 = full NoiseReducer + DeepFilterNet output (current/original behavior).
DEFAULT_NOISE_STRENGTH   = float(os.getenv("NOISE_REDUCTION_STRENGTH", "1.0"))

# ── Check FFmpeg availability ───────────────────────────
FFMPEG = shutil.which("ffmpeg")

if not FFMPEG:
    log.warning("FFmpeg not found in PATH. Falling back may fail.")

# ── Pipeline ────────────────────────────────────────────
sox_proc  = SoxProcessor(target_sr=SR)
nr_proc   = NoiseReducer()
enhancer  = StudioEnhancer(device="cuda" if os.getenv("CUDA", "0") == "1" else "cpu")
post_proc = PostProcessor(sr=SR)

executor = ThreadPoolExecutor(max_workers=2)
app = FastAPI(title="StudioEnhance — Hearing Aid Speech Enhancement")


# ────────────────────────────────────────────────────────
# AUDIO LOADER (FIXED + SAFE)
# ────────────────────────────────────────────────────────
async def _load_upload(audio: UploadFile):
    raw = await audio.read()

    # Try direct decode first (fast path)
    try:
        arr, sr = sf.read(io.BytesIO(raw))

        if arr.ndim > 1:
            arr = np.mean(arr, axis=1)

        return arr.astype(np.float32), sr

    except Exception:
        # fallback → ffmpeg required
        if not FFMPEG:
            raise HTTPException(
                status_code=500,
                detail="FFmpeg not installed or not found in PATH"
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as inp:
            inp.write(raw)
            input_path = inp.name

        output_path = input_path.replace(".m4a", ".wav")

        command = [
            FFMPEG,
            "-y",
            "-i",
            input_path,
            "-ac",
            "1",
            "-ar",
            str(SR),
            output_path
        ]

        try:
            subprocess.run(command, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise HTTPException(
                status_code=500,
                detail=f"FFmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}"
            )

        arr, sr = sf.read(output_path)

        if arr.ndim > 1:
            arr = np.mean(arr, axis=1)

        return arr.astype(np.float32), sr


# ────────────────────────────────────────────────────────
# AUDIO ENCODER
# ────────────────────────────────────────────────────────
def _to_wav(audio: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


# ────────────────────────────────────────────────────────
# ENHANCEMENT PIPELINE
# ────────────────────────────────────────────────────────
def _enhance(arr: np.ndarray, sr: int,
             voice_boost: bool = DEFAULT_VOICE_BOOST,
             boost_strength: float = DEFAULT_BOOST_STRENGTH,
             noise_strength: float = DEFAULT_NOISE_STRENGTH):
    arr, sr = sox_proc.process(arr, sr)
    dry     = arr  # pre-noise-reduction reference for the strength blend below
    arr     = nr_proc.process(arr, sr)
    arr     = enhancer.enhance(arr, sr=sr)

    if noise_strength < 1.0:
        n = min(len(dry), len(arr))
        arr = (noise_strength * arr[:n] + (1.0 - noise_strength) * dry[:n]).astype(np.float32)

    # Gate whatever DeepFilterNet left behind: speech passes through
    # untouched, everything else is pulled toward silence, scaled by
    # noise_strength (so 100% means near-complete elimination of gaps).
    arr = apply_noise_gate(arr, sr, strength=noise_strength)

    if voice_boost:
        arr = VoiceBoost(sr, strength=boost_strength).process(arr)
    arr     = post_proc.process(arr)
    return arr.astype(np.float32), sr


# ────────────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────────────
@app.get("/status")
@app.get("/health")
def status():
    return {
        "ok": True,
        "model": "SpeechBrain MetricGAN+ (pretrained)",
        "model_loaded": enhancer.is_loaded,
        "sample_rate": SR,
        "ffmpeg": bool(FFMPEG),
        "voice_boost_default": DEFAULT_VOICE_BOOST,
        "boost_strength_default": DEFAULT_BOOST_STRENGTH,
        "noise_strength_default": DEFAULT_NOISE_STRENGTH,
    }


@app.post("/filter")
async def filter_audio(
    audio: UploadFile = File(...),
    voice_boost: bool = Query(DEFAULT_VOICE_BOOST, description="Enable voice loudness enhancement"),
    boost_strength: float = Query(DEFAULT_BOOST_STRENGTH, ge=0.0, le=1.0,
                                   description="Voice boost strength, 0.0-1.0"),
    noise_strength: float = Query(DEFAULT_NOISE_STRENGTH, ge=0.0, le=1.0,
                                   description="Noise reduction strength, 0.0=off, 1.0=full"),
):
    arr, sr = await _load_upload(audio)

    loop = asyncio.get_event_loop()
    out, _ = await loop.run_in_executor(
        executor, _enhance, arr, sr, voice_boost, boost_strength, noise_strength)

    return Response(content=_to_wav(out), media_type="audio/wav")


@app.post("/enhance/wav")
async def enhance_wav(
    audio: UploadFile = File(...),
    voice_boost: bool = Query(DEFAULT_VOICE_BOOST),
    boost_strength: float = Query(DEFAULT_BOOST_STRENGTH, ge=0.0, le=1.0),
    noise_strength: float = Query(DEFAULT_NOISE_STRENGTH, ge=0.0, le=1.0),
):
    return await filter_audio(audio, voice_boost=voice_boost, boost_strength=boost_strength,
                               noise_strength=noise_strength)


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    params = websocket.query_params
    voice_boost = params.get("voice_boost", str(DEFAULT_VOICE_BOOST)).lower() == "true"
    boost_strength = float(params.get("boost_strength", DEFAULT_BOOST_STRENGTH))
    noise_strength = float(params.get("noise_strength", DEFAULT_NOISE_STRENGTH))

    voice_booster = VoiceBoost(SR, strength=boost_strength) if voice_boost else None
    await handle_stream(websocket, enhancer, post_proc, voice_booster, noise_strength)