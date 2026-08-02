"""
download_dataset.py — Free, lightweight training data fetcher

RECOMMENDED:  LibriSpeech 200 samples + locally-generated synthetic noise

  ~340 MB download (LibriSpeech dev-clean) — minutes, not hours
  Synthetic noise generated locally (white / pink / brown / bandpass)
       — no extra download, ~3 MB on disk
  Trains on CPU; total dataset on disk after cache cleanup: ~30 MB

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --samples 100 --noise-clips 50 --cleanup-cache
    python scripts/download_dataset.py --noise esc50    # +600 MB real noise
"""
from __future__ import annotations

import argparse
import logging
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-7s | %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('download')

SR        = 16_000
CLEAN_DIR = Path('data/clean')
NOISE_DIR = Path('data/noise')
CACHE_DIR = Path('data/.cache')

LIBRI_DEV_CLEAN_URL = 'https://www.openslr.org/resources/12/dev-clean.tar.gz'
ESC50_URL           = 'https://github.com/karoldvl/ESC-50/archive/master.zip'

_USER_AGENT = 'voice-filter-downloader/1.0'
_CHUNK      = 64 * 1024
_TIMEOUT    = 60


# ── Robust download with resume + retry ──────────────────────────────────────
def _content_length(url: str) -> int:
    try:
        req = urllib.request.Request(url, method='HEAD',
                                     headers={'User-Agent': _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return int(r.headers.get('Content-Length', 0))
    except Exception:
        return 0


def _download(url: str, dest: Path, label: str, max_retries: int = 6) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = _content_length(url)

    if dest.exists() and total > 0 and dest.stat().st_size >= total:
        log.info(f'  Already downloaded: {dest.name} ({dest.stat().st_size:,} B)')
        return

    log.info(f'  Downloading {label} ...')

    last_err = None
    for attempt in range(1, max_retries + 1):
        existing = dest.stat().st_size if dest.exists() else 0
        if total > 0 and existing >= total:
            log.info(f'  Saved -> {dest}')
            return

        headers = {'User-Agent': _USER_AGENT}
        if existing > 0:
            headers['Range'] = f'bytes={existing}-'
            log.info(f'  Resuming from byte {existing:,} (attempt {attempt}/{max_retries})')
        elif attempt > 1:
            log.info(f'  Retrying (attempt {attempt}/{max_retries})')

        req  = urllib.request.Request(url, headers=headers)
        mode = 'ab' if existing > 0 else 'wb'

        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                if total == 0:
                    cl = int(r.headers.get('Content-Length', 0))
                    if cl > 0:
                        total = cl + existing
                with open(dest, mode) as f, tqdm(
                        total=total or None, initial=existing,
                        unit='B', unit_scale=True, desc=label) as bar:
                    while True:
                        chunk = r.read(_CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        bar.update(len(chunk))

            done = dest.stat().st_size
            if total == 0 or done >= total:
                log.info(f'  Saved -> {dest} ({done:,} B)')
                return
            log.warning(f'  Short read: {done:,}/{total:,} B — will retry.')
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            last_err = e
            log.warning(f'  Attempt {attempt} failed: {type(e).__name__}: {e}')

        if attempt < max_retries:
            wait = 2 ** (attempt - 1)
            log.info(f'  Backing off {wait}s before next attempt ...')
            time.sleep(wait)

    raise RuntimeError(f'Download failed after {max_retries} attempts: {url}\n'
                       f'  Last error: {last_err}\n'
                       f'  Partial file kept at {dest} — re-run to resume.')


# ── LibriSpeech ──────────────────────────────────────────────────────────────
def download_librispeech(num_samples: int = 200) -> int:
    log.info('\n=== LibriSpeech dev-clean (recommended) ===')
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive = CACHE_DIR / 'LibriSpeech-dev-clean.tar.gz'
    _download(LIBRI_DEV_CLEAN_URL, archive, 'LibriSpeech dev-clean (~340 MB)')

    extract_dir = CACHE_DIR / 'LibriSpeech'
    if not extract_dir.exists():
        log.info('  Extracting tar.gz ...')
        with tarfile.open(archive, 'r:gz') as t:
            t.extractall(str(CACHE_DIR))

    flac_files = sorted(extract_dir.rglob('*.flac'))
    if not flac_files:
        log.error('  No .flac files found.')
        return 0

    by_speaker: dict[str, list[Path]] = {}
    for f in flac_files:
        spk = f.parent.parent.name
        by_speaker.setdefault(spk, []).append(f)

    speakers = list(by_speaker.keys())
    picked: list[Path] = []
    while len(picked) < num_samples and any(by_speaker.values()):
        for spk in speakers:
            if by_speaker[spk]:
                picked.append(by_speaker[spk].pop(0))
                if len(picked) >= num_samples:
                    break

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for src in tqdm(picked, desc='Resample speech'):
        try:
            audio, _ = librosa.load(str(src), sr=SR, mono=True)
            sf.write(str(CLEAN_DIR / f'libri_{src.stem}.wav'),
                     audio, SR, subtype='PCM_16')
            saved += 1
        except Exception as e:
            log.warning(f'  Skipped {src.name}: {e}')

    log.info(f'  Saved {saved} files -> {CLEAN_DIR}')
    return saved


# ── Synthetic noise (no download) ────────────────────────────────────────────
def _shaped_noise(kind: str, n: int, sr: int) -> np.ndarray:
    rng   = np.random.default_rng()
    white = rng.standard_normal(n).astype(np.float32)

    if kind == 'white':
        out = white
    elif kind in ('pink', 'brown'):
        spec  = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        freqs[0] = 1.0
        scale = np.sqrt(freqs) if kind == 'pink' else freqs
        out   = np.fft.irfft(spec / scale, n=n).astype(np.float32)
    elif kind == 'bandpass':
        from scipy.signal import butter, sosfilt
        low  = float(rng.uniform(150.0,  800.0))
        high = float(rng.uniform(2_000.0, 4_500.0))
        sos  = butter(4, [low, high], btype='bandpass', fs=sr, output='sos')
        out  = sosfilt(sos, white).astype(np.float32)
    else:
        raise ValueError(f'unknown noise kind: {kind}')

    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.7).astype(np.float32)


def generate_synthetic_noise(num_clips: int = 100, duration_sec: float = 5.0) -> int:
    log.info('\n=== Synthetic noise generator (no download) ===')
    NOISE_DIR.mkdir(parents=True, exist_ok=True)

    n_samples = int(SR * duration_sec)
    kinds     = ['white', 'pink', 'brown', 'bandpass']

    saved = 0
    for i in tqdm(range(num_clips), desc='Synth noise'):
        kind = kinds[i % len(kinds)]
        try:
            audio = _shaped_noise(kind, n_samples, SR)
            sf.write(str(NOISE_DIR / f'synth_{kind}_{i:03d}.wav'),
                     audio, SR, subtype='PCM_16')
            saved += 1
        except Exception as e:
            log.warning(f'  Skipped synth #{i} ({kind}): {e}')

    log.info(f'  Generated {saved} synthetic noise clips -> {NOISE_DIR}')
    return saved


# ── ESC-50 (opt-in) ──────────────────────────────────────────────────────────
def download_esc50() -> int:
    log.info('\n=== ESC-50 Environmental Sound Dataset (opt-in) ===')
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / 'ESC50.zip'
    _download(ESC50_URL, zip_path, 'ESC-50 (~600 MB)')

    extract_dir = CACHE_DIR / 'ESC50'
    if not extract_dir.exists():
        log.info('  Extracting ...')
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(str(extract_dir))

    NOISE_DIR.mkdir(parents=True, exist_ok=True)
    wavs = list(extract_dir.rglob('*.wav')) + list(extract_dir.rglob('*.ogg'))

    saved = 0
    for src in tqdm(wavs, desc='Resample noise'):
        try:
            audio, _ = librosa.load(str(src), sr=SR, mono=True)
            sf.write(str(NOISE_DIR / f'esc50_{src.stem}.wav'),
                     audio, SR, subtype='PCM_16')
            saved += 1
        except Exception as e:
            log.warning(f'  Skipped {src.name}: {e}')

    log.info(f'  Saved {saved} noise files -> {NOISE_DIR}')
    return saved


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=int, default=200)
    p.add_argument('--noise', default='synthetic',
                   choices=['synthetic', 'esc50', 'none'])
    p.add_argument('--noise-clips', type=int, default=100)
    p.add_argument('--cleanup-cache', action='store_true')
    args = p.parse_args()

    log.info('=' * 60)
    log.info('VoiceFilter Dataset Downloader (lightweight default)')
    log.info('=' * 60)

    n_speech = download_librispeech(num_samples=max(50, min(args.samples, 5000)))

    if args.noise == 'synthetic':
        n_noise = generate_synthetic_noise(num_clips=args.noise_clips)
    elif args.noise == 'esc50':
        n_noise = download_esc50()
    else:
        n_noise = 0

    if args.cleanup_cache and CACHE_DIR.exists():
        log.info(f'\nRemoving cache {CACHE_DIR} ...')
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    log.info('\n' + '=' * 60)
    log.info('Download complete!')
    log.info(f'  Clean speech : {n_speech} files -> {CLEAN_DIR}')
    log.info(f'  Noise clips  : {n_noise} files -> {NOISE_DIR}')
    log.info('=' * 60)
    log.info('\nNext: python train.py')


if __name__ == '__main__':
    main()
