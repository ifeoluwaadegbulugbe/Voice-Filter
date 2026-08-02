"""
download_diverse_voices.py — Harvest diverse-accent speech from public corpora.

Sources used (all freely available, no commercial use):
  1. OpenSLR-70  Nigerian English   ~1.2 GB direct download, no auth
  2. AfriSpeech-200 (Pan-African)    streamed from Hugging Face (no full download)
  3. TED-LIUM (international English) streamed from Hugging Face
  4. (optional) Mozilla Common Voice English subset — needs HF login

Output goes to data/clean/ as 16 kHz mono PCM_16 WAVs, ready for train.py.

Default behavior (no args):
    Downloads OpenSLR-70 Nigerian English in full (1.2 GB), then streams
    300 clips from AfriSpeech (50 from each of 6 accents). Total: ~3000 clips,
    ~1.5 GB on disk.

Usage:
    python scripts/download_diverse_voices.py
    python scripts/download_diverse_voices.py --skip-openslr      # no Nigerian dataset
    python scripts/download_diverse_voices.py --afrispeech-per-accent 100
    python scripts/download_diverse_voices.py --tedlium 200       # add 200 TED clips
    python scripts/download_diverse_voices.py --common-voice 300  # adds CV (requires HF login)

LICENSE / FAIR USE
==================
  OpenSLR-70    : Apache-2.0
  AfriSpeech    : CC BY-NC-SA 4.0  (non-commercial)
  TED-LIUM      : CC BY-NC-ND 3.0  (non-commercial)
  Common Voice  : CC0
Use locally for training your own model. Do not redistribute the audio.
"""
from __future__ import annotations

import argparse
import io
import logging
import shutil
import sys
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
log = logging.getLogger('diverse')

SR        = 16_000
CLEAN_DIR = Path('data/clean')
CACHE_DIR = Path('data/.cache')

OPENSLR_70_FEMALE = 'https://www.openslr.org/resources/70/en_ng_female.zip'
OPENSLR_70_MALE   = 'https://www.openslr.org/resources/70/en_ng_male.zip'

# Accents to sample from AfriSpeech-200 (top Pan-African).
AFRISPEECH_ACCENTS = [
    'yoruba',     # Nigeria
    'igbo',       # Nigeria
    'hausa',      # Nigeria / Niger
    'swahili',    # East Africa
    'isizulu',    # South Africa
    'twi',        # Ghana
]

# ── Robust download with resume / retry (same as download_dataset.py) ────────
_USER_AGENT = 'voice-filter-downloader/1.0'
_CHUNK      = 64 * 1024
_TIMEOUT    = 60


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
            return
        headers = {'User-Agent': _USER_AGENT}
        if existing > 0:
            headers['Range'] = f'bytes={existing}-'
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
                        if not chunk: break
                        f.write(chunk); bar.update(len(chunk))
            if total == 0 or dest.stat().st_size >= total:
                log.info(f'  Saved -> {dest}')
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            last_err = e
            log.warning(f'  Attempt {attempt} failed: {type(e).__name__}: {e}')
        if attempt < max_retries:
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f'Download failed after {max_retries} attempts: {url}\n  Last: {last_err}')


# ──────────────────────────────────────────────────────────────────────────────
#  1. OpenSLR-70 Nigerian English
# ──────────────────────────────────────────────────────────────────────────────
def fetch_openslr_70(out_dir: Path) -> int:
    log.info('\n=== OpenSLR-70 Nigerian English ===')
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for url, label in [(OPENSLR_70_FEMALE, 'NG English female (~759 MB)'),
                       (OPENSLR_70_MALE,   'NG English male   (~454 MB)')]:
        zip_path = CACHE_DIR / Path(url).name
        try:
            _download(url, zip_path, label)
        except Exception as e:
            log.error(f'  Could not download {label}: {e}')
            continue

        log.info(f'  Extracting {zip_path.name} ...')
        with zipfile.ZipFile(zip_path, 'r') as z:
            members = [m for m in z.namelist() if m.lower().endswith('.wav')]
            for m in tqdm(members, desc=f'  resample {zip_path.stem}'):
                try:
                    with z.open(m) as f:
                        raw = f.read()
                    audio, _ = librosa.load(io.BytesIO(raw), sr=SR, mono=True)
                    name = f'openslr70_{Path(m).stem}.wav'
                    sf.write(str(out_dir / name), audio, SR, subtype='PCM_16')
                    saved += 1
                except Exception as e:
                    log.warning(f'  skipped {m}: {e}')
    log.info(f'  Saved {saved} OpenSLR-70 files -> {out_dir}')
    return saved


# ──────────────────────────────────────────────────────────────────────────────
#  2. AfriSpeech-200 (streaming, Pan-African)
# ──────────────────────────────────────────────────────────────────────────────
def fetch_afrispeech(out_dir: Path, per_accent: int = 50,
                     accents: list[str] = AFRISPEECH_ACCENTS) -> int:
    log.info('\n=== AfriSpeech-200 (streaming) ===')
    try:
        from datasets import load_dataset
    except ImportError:
        log.error('  pip install datasets   to use AfriSpeech.')
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    total_saved = 0
    for accent in accents:
        log.info(f'  streaming {per_accent} clips of accent={accent} ...')
        try:
            ds = load_dataset('intronhealth/afrispeech-200', accent,
                              split='train', streaming=True)
        except Exception as e:
            log.warning(f'  skipping {accent}: {e}')
            continue

        n = 0
        for sample in ds:
            if n >= per_accent: break
            try:
                audio_arr = np.asarray(sample['audio']['array'], dtype=np.float32)
                src_sr    = int(sample['audio']['sampling_rate'])
                if src_sr != SR:
                    audio_arr = librosa.resample(audio_arr, orig_sr=src_sr, target_sr=SR)
                # Peak-normalise to -3 dBFS to match the rest of the corpus.
                peak = float(np.max(np.abs(audio_arr))) or 1.0
                audio_arr = (audio_arr / peak * 0.71).astype(np.float32)

                clip_id = sample.get('audio_id') or f'{accent}_{n:04d}'
                name    = f'afri_{accent}_{clip_id}.wav'.replace('/', '_')
                sf.write(str(out_dir / name), audio_arr, SR, subtype='PCM_16')
                n += 1
            except Exception as e:
                log.warning(f'  bad sample in {accent}: {e}')
        total_saved += n
        log.info(f'    -> {n} clips saved for {accent}')
    log.info(f'  Saved {total_saved} AfriSpeech files -> {out_dir}')
    return total_saved


# ──────────────────────────────────────────────────────────────────────────────
#  3. TED-LIUM (streaming, international English)
# ──────────────────────────────────────────────────────────────────────────────
def fetch_tedlium(out_dir: Path, num_clips: int = 200) -> int:
    log.info('\n=== TED-LIUM (streaming) ===')
    try:
        from datasets import load_dataset
    except ImportError:
        log.error('  pip install datasets   to use TED-LIUM.')
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f'  streaming {num_clips} TED-LIUM clips ...')
    try:
        ds = load_dataset('LIUM/tedlium', 'release3',
                          split='train', streaming=True,
                          trust_remote_code=True)
    except Exception as e:
        log.error(f'  Could not load TED-LIUM: {e}')
        return 0

    saved = 0
    for sample in ds:
        if saved >= num_clips: break
        try:
            audio_arr = np.asarray(sample['audio']['array'], dtype=np.float32)
            src_sr    = int(sample['audio']['sampling_rate'])
            if src_sr != SR:
                audio_arr = librosa.resample(audio_arr, orig_sr=src_sr, target_sr=SR)
            # TED clips can be very long — cap at 15 s for training.
            if len(audio_arr) > SR * 15:
                audio_arr = audio_arr[:SR * 15]
            elif len(audio_arr) < SR * 1:
                continue                                         # too short
            peak = float(np.max(np.abs(audio_arr))) or 1.0
            audio_arr = (audio_arr / peak * 0.71).astype(np.float32)

            clip_id = sample.get('id') or f'ted_{saved:04d}'
            name    = f'ted_{clip_id}.wav'.replace('/', '_')
            sf.write(str(out_dir / name), audio_arr, SR, subtype='PCM_16')
            saved += 1
            if saved % 50 == 0:
                log.info(f'    {saved}/{num_clips} ...')
        except Exception as e:
            log.warning(f'  bad sample: {e}')
    log.info(f'  Saved {saved} TED-LIUM files -> {out_dir}')
    return saved


# ──────────────────────────────────────────────────────────────────────────────
#  4. Mozilla Common Voice English (optional, requires HF login)
# ──────────────────────────────────────────────────────────────────────────────
def fetch_common_voice(out_dir: Path, num_clips: int = 300,
                       accent_filter: list[str] | None = None) -> int:
    log.info('\n=== Mozilla Common Voice English (streaming) ===')
    log.info('  NOTE: Requires `huggingface-cli login` once. Free account.')
    try:
        from datasets import load_dataset
    except ImportError:
        log.error('  pip install datasets   to use Common Voice.')
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_dataset('mozilla-foundation/common_voice_17_0', 'en',
                          split='train', streaming=True)
    except Exception as e:
        log.error(f'  Could not load Common Voice (login first?): {e}')
        return 0

    saved = 0
    for sample in ds:
        if saved >= num_clips: break
        if accent_filter:
            acc = (sample.get('accent') or '').lower()
            if not any(a.lower() in acc for a in accent_filter):
                continue
        try:
            audio_arr = np.asarray(sample['audio']['array'], dtype=np.float32)
            src_sr    = int(sample['audio']['sampling_rate'])
            if src_sr != SR:
                audio_arr = librosa.resample(audio_arr, orig_sr=src_sr, target_sr=SR)
            peak = float(np.max(np.abs(audio_arr))) or 1.0
            audio_arr = (audio_arr / peak * 0.71).astype(np.float32)

            client_id = (sample.get('client_id') or 'cv')[:8]
            name = f'cv_{client_id}_{saved:04d}.wav'
            sf.write(str(out_dir / name), audio_arr, SR, subtype='PCM_16')
            saved += 1
        except Exception as e:
            log.warning(f'  bad sample: {e}')
    log.info(f'  Saved {saved} Common Voice files -> {out_dir}')
    return saved


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', default=str(CLEAN_DIR))
    p.add_argument('--skip-openslr', action='store_true',
                   help='Skip the 1.2 GB Nigerian English dataset.')
    p.add_argument('--skip-afrispeech', action='store_true',
                   help='Skip AfriSpeech streaming.')
    p.add_argument('--afrispeech-per-accent', type=int, default=50,
                   help='How many clips to stream per AfriSpeech accent (default 50).')
    p.add_argument('--tedlium', type=int, default=0,
                   help='Stream this many TED-LIUM clips (default 0 = skip).')
    p.add_argument('--common-voice', type=int, default=0,
                   help='Stream this many Common Voice clips (default 0 = skip). '
                        'Requires `huggingface-cli login` first.')
    p.add_argument('--cv-accents', nargs='*', default=None,
                   help='Filter Common Voice by accent keywords (e.g. african indian filipino).')
    p.add_argument('--cleanup-cache', action='store_true',
                   help='Delete data/.cache/ when done.')
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    log.info('=' * 60)
    log.info('Diverse Voice Harvester')
    log.info('=' * 60)

    totals = {}

    if not args.skip_openslr:
        try:
            totals['openslr_70'] = fetch_openslr_70(out_dir)
        except Exception as e:
            log.error(f'OpenSLR-70 failed: {e}')
            totals['openslr_70'] = 0

    if not args.skip_afrispeech:
        try:
            totals['afrispeech'] = fetch_afrispeech(
                out_dir, per_accent=args.afrispeech_per_accent)
        except Exception as e:
            log.error(f'AfriSpeech failed: {e}')
            totals['afrispeech'] = 0

    if args.tedlium > 0:
        try:
            totals['tedlium'] = fetch_tedlium(out_dir, num_clips=args.tedlium)
        except Exception as e:
            log.error(f'TED-LIUM failed: {e}')
            totals['tedlium'] = 0

    if args.common_voice > 0:
        try:
            totals['common_voice'] = fetch_common_voice(
                out_dir, num_clips=args.common_voice, accent_filter=args.cv_accents)
        except Exception as e:
            log.error(f'Common Voice failed: {e}')
            totals['common_voice'] = 0

    if args.cleanup_cache and CACHE_DIR.exists():
        log.info(f'\nCleaning cache {CACHE_DIR} ...')
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    log.info('\n' + '=' * 60)
    log.info('Done. Per-source totals:')
    for k, v in totals.items():
        log.info(f'  {k:<14} {v:>6} files')
    log.info(f'  {"TOTAL":<14} {sum(totals.values()):>6} files -> {out_dir}')
    log.info('Next: python train.py')


if __name__ == '__main__':
    main()
