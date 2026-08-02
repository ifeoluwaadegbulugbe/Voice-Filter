"""
ws_handler.py — Real-time WebSocket audio streaming.

Protocol (raw int16 PCM in both directions):
  * Client sends raw int16 PCM bytes, 16 kHz mono
  * Server accumulates 300 ms before processing for better Transformer context
  * Server sends back enhanced int16 PCM the moment a chunk is ready
"""
from __future__ import annotations

import asyncio
import logging

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from ai_pipeline.noise_gate import apply_noise_gate

logger = logging.getLogger('ws_handler')

SR              = 16_000
PROCESS_MS      = 300
PROCESS_SAMPLES = int(SR * PROCESS_MS / 1000)        # 4 800 samples
PROCESS_BYTES   = PROCESS_SAMPLES * 2                # int16 = 2 bytes


def _process_chunk(pcm_bytes: bytes, enhancer, post_proc, voice_booster,
                    noise_strength: float = 1.0) -> bytes:
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    enhanced = enhancer.enhance(audio, sr=SR)

    if noise_strength < 1.0:
        n = min(len(audio), len(enhanced))
        enhanced = (noise_strength * enhanced[:n] + (1.0 - noise_strength) * audio[:n]).astype(np.float32)

    enhanced = apply_noise_gate(enhanced, SR, strength=noise_strength,
                                 hold_ms=60.0, release_ms=80.0)  # shorter for 300ms chunks

    if voice_booster is not None:
        # Stateful: envelope/loudness estimate carries across chunks so gain
        # doesn't pump or click at 300 ms chunk boundaries.
        enhanced = voice_booster.process(enhanced)
    enhanced = post_proc.process(enhanced)
    out16    = np.clip(enhanced * 32767.0, -32768, 32767).astype(np.int16)
    return out16.tobytes()


async def handle_stream(websocket: WebSocket, enhancer, post_proc, voice_booster=None,
                         noise_strength: float = 1.0) -> None:
    await websocket.accept()
    logger.info('WebSocket /ws/stream connected.')

    buffer = bytearray()
    loop   = asyncio.get_event_loop()

    try:
        while True:
            data = await websocket.receive_bytes()
            buffer.extend(data)

            # Flush whole 300 ms chunks while we have enough.
            while len(buffer) >= PROCESS_BYTES:
                chunk = bytes(buffer[:PROCESS_BYTES])
                del buffer[:PROCESS_BYTES]
                enhanced = await loop.run_in_executor(
                    None, _process_chunk, chunk, enhancer, post_proc, voice_booster, noise_strength)
                await websocket.send_bytes(enhanced)
    except WebSocketDisconnect:
        logger.info('WebSocket disconnected.')
    except Exception as e:
        logger.error(f'WebSocket error: {e}')
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
