"""
Pre-download all pretrained model weights.
Run ONCE after pip install.

Usage:
    python scripts/download_models.py
"""
import os

from dotenv import load_dotenv

load_dotenv()
HF_TOKEN   = os.getenv('HF_TOKEN', '')
MODEL_PATH = os.getenv('MODEL_PATH', 'checkpoints/best_model.pth')

print('\n=== Studio Speech Enhancer — Model Setup ===\n')

# ── 1. Trained checkpoint ────────────────────────────────────────────────────
print('[1/3] StudioEnhanceNet trained checkpoint ...')
if os.path.exists(MODEL_PATH):
    size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
    print(f'  OK — found {MODEL_PATH}  ({size_mb:.1f} MB)')
else:
    print(f'  MISSING — checkpoint not found at {MODEL_PATH}')
    print('  -> Train first:  python train.py')

# ── 2. pyannote VAD (optional) ───────────────────────────────────────────────
print('\n[2/3] pyannote Voice Activity Detection model ...')
if not HF_TOKEN.startswith('hf_') or len(HF_TOKEN) < 10:
    print('  SKIP — HF_TOKEN not set. VAD will be disabled.')
    print('  -> Get a token at https://huggingface.co/settings/tokens')
else:
    try:
        from pyannote.audio import Pipeline
        Pipeline.from_pretrained('pyannote/voice-activity-detection', token=HF_TOKEN)
        print('  OK — pyannote VAD ready')
    except Exception as e:
        print(f'  FAIL: {e}')

# ── 3. ECAPA-TDNN (optional) ─────────────────────────────────────────────────
print('\n[3/3] SpeechBrain ECAPA-TDNN (optional identity verifier) ...')
try:
    os.environ.setdefault('SB_DISABLE_K2', '1')
    os.environ.setdefault('K2_DISABLE',    '1')
    from speechbrain.inference import EncoderClassifier
    EncoderClassifier.from_hparams(
        source   = 'speechbrain/spkrec-ecapa-voxceleb',
        run_opts = {'device': 'cpu'},
    )
    print('  OK — ECAPA-TDNN ready')
except Exception as e:
    print(f'  FAIL (non-critical): {e}')

print('\n' + '=' * 50)
print('Setup complete.')
print('Start the server:')
print('  uvicorn backend.server:app --host 0.0.0.0 --port 8000')
