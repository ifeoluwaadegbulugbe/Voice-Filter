# Voice-Filter

Final Year Project — AI-powered speech enhancement app for hearing-aid-style
noise suppression. Combines [DeepFilterNet 3](https://github.com/Rikorose/DeepFilterNet)
with an adjustable noise gate and a voice loudness booster, served through a
FastAPI backend and a Flutter client (desktop + mobile).

## What it does

Given a noisy recording, the pipeline:

1. **Preprocesses** — mono conversion, resampling, DC removal, peak normalization
2. **Reduces noise** — spectral subtraction (`noisereduce`) as a first pass
3. **Enhances** — [DeepFilterNet 3](https://github.com/Rikorose/DeepFilterNet) removes background noise while preserving speech, with a SpeechBrain MetricGAN+ fallback if DeepFilterNet is unavailable
4. **Gates** — an energy-based VAD noise gate pulls whatever's left in non-speech regions down toward silence, strength adjustable from the client
5. **Boosts** — a LUFS-aware loudness leveler, soft-knee compressor, and brick-wall limiter make quiet speech easier to hear without amplifying residual noise or clipping
6. **Limits & normalizes** — final peak-safe limiter before export

Both a one-shot file endpoint and a real-time WebSocket stream are supported.

## Project structure

```
backend/            FastAPI server — REST + WebSocket endpoints
ai_pipeline/         DeepFilterNet enhancer, noise gate, voice boost, post-processing
preprocessing/       Input normalization and spectral-subtraction noise reduction
mobile/              Flutter app (Windows/Android/iOS/web) — record, upload, tune, playback
src/                 Custom StudioEnhanceNet model + training pipeline
scripts/             Dataset download, benchmarking, and debugging utilities
train.py             Model training entry point
evaluate.py          Evaluation entry point
```

## Backend

**Requirements:** Python 3.11, [FFmpeg](https://ffmpeg.org/) on `PATH`.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```

### API

| Endpoint | Method | Description |
|---|---|---|
| `/status`, `/health` | GET | Health check + pipeline defaults |
| `/filter` | POST | Upload audio, get enhanced WAV back |
| `/enhance/wav` | POST | Alias for `/filter` |
| `/ws/stream` | WS | Real-time 16 kHz PCM streaming |

Query params on `/filter` and `/ws/stream`:

- `voice_boost` (bool, default `true`) / `boost_strength` (0.0–1.0, default `0.7`)
- `noise_strength` (0.0–1.0, default `1.0`) — how aggressively the noise gate + DeepFilterNet blend is applied

## Mobile app

**Requirements:** Flutter SDK (>=3.0.0).

```bash
cd mobile
flutter pub get
flutter run -d windows      # or android / ios / chrome
```

Set the backend URL in the app's Settings screen (defaults to
`http://10.0.2.2:8000` for the Android emulator; use your machine's actual
address for physical devices or desktop). Record or upload audio, then use
the **Enhance Speech** slider on the result screen to tune noise-reduction
strength live against that same recording.

## Tech stack

- **Backend:** FastAPI, PyTorch, DeepFilterNet, librosa, soundfile, pyloudnorm, noisereduce
- **Mobile:** Flutter, `record`, `audioplayers`, `file_picker`, `shared_preferences`
