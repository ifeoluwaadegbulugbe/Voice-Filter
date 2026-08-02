"""
Convert any folder of audio files to 16 kHz mono WAVs.
Useful for prepping your own recordings before training.

Usage:
    python convert_to_wav.py
    python convert_to_wav.py --input my_audio --output data/raw/clean
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import librosa
import soundfile as sf


def convert_folder(input_folder: str, output_folder: str, sr: int = 16_000) -> int:
    os.makedirs(output_folder, exist_ok=True)
    extensions = ['*.m4a', '*.mp3', '*.mp4', '*.aac', '*.ogg', '*.wav', '*.flac']
    files = []
    for ext in extensions:
        files.extend(Path(input_folder).glob(ext))

    if not files:
        print(f'No audio files in {input_folder}')
        return 0

    print(f'Found {len(files)} files in {input_folder}')
    converted = 0
    for i, src in enumerate(files, 1):
        try:
            audio, _ = librosa.load(str(src), sr=sr, mono=True)
            dst = Path(output_folder) / f'{src.stem}.wav'
            sf.write(str(dst), audio, sr, subtype='PCM_16')
            print(f'  [{i}/{len(files)}] {src.name} -> {dst.name}')
            converted += 1
        except Exception as e:
            print(f'  [{i}/{len(files)}] ERROR {src.name}: {e}')
    return converted


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input',  default='data/raw/clean')
    p.add_argument('--output', default='data/raw/clean')
    p.add_argument('--sr',     type=int, default=16_000)
    args = p.parse_args()

    n = convert_folder(args.input, args.output, args.sr)
    print(f'\nDone. {n} files converted -> {args.output}')
