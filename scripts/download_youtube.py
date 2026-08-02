"""
download_youtube.py — Harvest training audio from YouTube channels / playlists.

Use this to broaden the accent / dialect coverage of your training set
beyond what LibriSpeech provides. Particularly useful for Nigerian, African,
South Asian, and Caribbean English which are under-represented in
academic speech corpora.

Output: 16 kHz mono PCM_16 WAV files in data/clean/ (or --output-dir).
Each download is converted on the fly via ffmpeg (bundled with
imageio-ffmpeg, no system install required).

Usage:
    # One channel — first 20 episodes
    python scripts/download_youtube.py https://www.youtube.com/@ISaidWhatISaidPod --max 20

    # Several channels at once
    python scripts/download_youtube.py URL1 URL2 URL3 --max 10

    # From a text file (one URL per line)
    python scripts/download_youtube.py --from-file my_channels.txt --max 10

    # Pre-split each episode into 10-second clips for better training diversity
    python scripts/download_youtube.py URL --max 5 --split-sec 10

LEGAL NOTE
==========
Using YouTube audio for personal model training is generally treated as
fair use (academic / non-commercial). Do NOT redistribute the downloaded
audio or any derivative dataset publicly without the creators' permission.
This script is for your local prototype — keep it that way.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-7s | %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('youtube')

SR        = 16_000
CLEAN_DIR = Path('data/clean')
TMP_DIR   = Path('data/.cache/youtube')


# ──────────────────────────────────────────────────────────────────────────────
def _check_dependencies():
    missing = []
    try:
        import yt_dlp        # noqa: F401
    except ImportError:
        missing.append('yt-dlp')
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        missing.append('imageio-ffmpeg')
    if missing:
        log.error('Missing packages: ' + ', '.join(missing))
        log.error('Install with:  pip install ' + ' '.join(missing))
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
def download_audio(urls: list[str], max_per_url: int, out_dir: Path) -> list[Path]:
    """
    Pull audio-only from each URL, convert to WAV via ffmpeg, drop into
    `out_dir`. Returns list of resulting WAV paths.
    """
    import imageio_ffmpeg
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    archive = out_dir / 'download_archive.txt'    # avoid re-downloading

    ydl_opts = {
        'format':            'bestaudio/best',
        'outtmpl':           str(out_dir / '%(uploader)s__%(title).80B__%(id)s.%(ext)s'),
        'restrictfilenames': True,
        'noplaylist':        False,
        'playlistend':       max_per_url,
        'quiet':             False,
        'no_warnings':       True,
        'ignoreerrors':      True,
        'download_archive':  str(archive),
        'ffmpeg_location': ffmpeg_path,
        'postprocessors': [{
            'key':              'FFmpegExtractAudio',
            'preferredcodec':   'wav',
            'preferredquality': '0',
        }],
        'postprocessor_args': ['-ar', str(SR), '-ac', '1'],
    }

    log.info(f'ffmpeg: {ffmpeg_path}')
    log.info(f'archive (skip-list): {archive}')

    before = {p.resolve() for p in out_dir.glob('*.wav')}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            log.info(f'  downloading: {url}  (max {max_per_url})')
            try:
                ydl.download([url])
            except Exception as e:
                log.error(f'  FAILED {url}: {e}')

    after = {p.resolve() for p in out_dir.glob('*.wav')}
    new   = sorted(after - before)
    log.info(f'  -> {len(new)} new WAV file(s)')
    return new


# ──────────────────────────────────────────────────────────────────────────────
def split_into_chunks(src_wav: Path, dst_dir: Path,
                       chunk_sec: float = 10.0,
                       skip_intro_sec: float = 30.0,
                       skip_outro_sec: float = 30.0) -> int:
    """
    Cut a long episode into shorter clips, skipping the first and last
    `skip_*_sec` seconds (typical intro/outro music + ads).
    """
    audio, _ = librosa.load(str(src_wav), sr=SR, mono=True)
    if len(audio) <= int(SR * (skip_intro_sec + skip_outro_sec + chunk_sec)):
        return 0

    start = int(SR * skip_intro_sec)
    end   = max(start, len(audio) - int(SR * skip_outro_sec))
    span  = audio[start:end]

    chunk_len = int(SR * chunk_sec)
    n_chunks  = len(span) // chunk_len

    saved = 0
    for i in range(n_chunks):
        clip = span[i * chunk_len: (i + 1) * chunk_len]
        peak = float(abs(clip).max())
        if peak < 0.01:                    # mostly silence -> skip
            continue
        out_name = f'{src_wav.stem}__chunk{i:03d}.wav'
        sf.write(str(dst_dir / out_name), clip, SR, subtype='PCM_16')
        saved += 1
    return saved


# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('urls', nargs='*', help='YouTube channel / playlist / video URLs')
    p.add_argument('--from-file', type=str, default=None,
                   help='Text file with one URL per line.')
    p.add_argument('--max', type=int, default=10,
                   help='Max videos per URL (default: 10).')
    p.add_argument('--output-dir', default=str(CLEAN_DIR),
                   help='Where final WAVs land (default: data/clean/).')
    p.add_argument('--cache-dir', default=str(TMP_DIR),
                   help='Where downloads go before processing.')
    p.add_argument('--split-sec', type=float, default=0.0,
                   help='If > 0, split each episode into N-sec clips (recommended: 10).')
    p.add_argument('--skip-intro', type=float, default=30.0,
                   help='Seconds to skip from the start (intro music). Default 30.')
    p.add_argument('--skip-outro', type=float, default=30.0,
                   help='Seconds to skip from the end (outro music + ads). Default 30.')
    p.add_argument('--keep-cache', action='store_true',
                   help='Keep the cache directory after success.')
    args = p.parse_args()

    _check_dependencies()

    urls = list(args.urls)
    if args.from_file:
        urls.extend([
            u.strip()
            for u in Path(args.from_file).read_text(encoding="utf-8", errors="ignore").splitlines()
            if u.strip() and not u.strip().startswith('#')
        ])
    if not urls:
        log.error('No URLs provided.  Pass them on the command line or via --from-file.')
        sys.exit(1)

    cache_dir  = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Pull audio
    new_wavs = download_audio(urls, args.max, cache_dir)

    if not new_wavs:
        log.warning('No new audio downloaded (already cached?).')
        # Still try to (re)split everything currently in cache.
        new_wavs = sorted(cache_dir.glob('*.wav'))

    # 2) Either copy whole episodes or split into smaller training clips.
    if args.split_sec > 0:
        log.info(f'Splitting episodes into {args.split_sec:.0f}-second clips ...')
        total_chunks = 0
        for src in tqdm(new_wavs, desc='split'):
            n = split_into_chunks(src, output_dir,
                                  chunk_sec      = args.split_sec,
                                  skip_intro_sec = args.skip_intro,
                                  skip_outro_sec = args.skip_outro)
            total_chunks += n
        log.info(f'  -> {total_chunks} training clips written to {output_dir}/')
    else:
        log.info(f'Copying whole episodes to {output_dir}/ ...')
        for src in new_wavs:
            shutil.copy2(src, output_dir / src.name)
        log.info(f'  -> {len(new_wavs)} files copied.')

    # 3) Cleanup
    if not args.keep_cache and cache_dir.exists():
        log.info(f'Removing cache {cache_dir} ...')
        shutil.rmtree(cache_dir, ignore_errors=True)

    log.info('Done. Train with:  python train.py')


if __name__ == '__main__':
    main()
