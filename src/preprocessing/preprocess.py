"""
preprocess.py — Convert raw audio into 16 kHz mono PCM_16 WAVs

INPUTS:
    data/clean/
    data/.cache/

OUTPUTS:
    data/clean/   (processed)
    data/noise/   (processed)

Usage:
    python src/preprocessing/preprocess.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import librosa
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("preprocess")

SR = 16_000


# ─────────────────────────────────────────────────────────────
def _convert_folder(src: Path, dst: Path, label: str) -> int:
    if not src.exists():
        log.warning(f"{label}: {src} not found — skipping.")
        return 0

    dst.mkdir(parents=True, exist_ok=True)

    audio_files = []
    exts = ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.ogg", "*.aac")

    for ext in exts:
        audio_files.extend(src.rglob(ext))

    if not audio_files:
        log.warning(f"{label}: no audio files in {src}")
        return 0

    saved = 0

    for f in tqdm(audio_files, desc=label):
        try:
            audio, _ = librosa.load(str(f), sr=SR, mono=True)

            # skip empty audio
            if len(audio) < SR * 1:
                continue

            out_path = dst / f"{f.stem}.wav"
            sf.write(str(out_path), audio, SR, subtype="PCM_16")
            saved += 1

        except Exception as e:
            log.warning(f"Skipped {f.name}: {e}")

    log.info(f"{label}: {saved} files → {dst}")
    return saved


# ─────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()

    # UPDATED INPUT SOURCES
    p.add_argument("--clean-src", default="data/clean")
    p.add_argument("--cache-src", default="data/.cache")
    p.add_argument("--noise-src", default="data/noise")

    # OUTPUTS
    p.add_argument("--clean-out", default="data/clean")
    p.add_argument("--noise-out", default="data/noise")

    args = p.parse_args()

    log.info("=== Preprocessing Audio → 16kHz Mono WAV ===")

    clean_src = Path(args.clean_src)
    cache_src = Path(args.cache_src)
    noise_src = Path(args.noise_src)

    clean_out = Path(args.clean_out)
    noise_out = Path(args.noise_out)

    # ── CLEAN SPEECH ─────────────────────────────
    n_clean = _convert_folder(clean_src, clean_out, "clean")

    # ── CACHE (extra speech pool) ────────────────
    n_cache = _convert_folder(cache_src, clean_out, "cache-speech")

    # ── NOISE ─────────────────────────────────────
    n_noise = _convert_folder(noise_src, noise_out, "noise")

    log.info("=" * 60)
    log.info(f"Done:")
    log.info(f"  Clean speech : {n_clean + n_cache}")
    log.info(f"  Noise        : {n_noise}")
    log.info("Next: python train.py")


if __name__ == "__main__":
    main()