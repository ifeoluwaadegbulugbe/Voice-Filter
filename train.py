"""
train.py — StudioEnhanceNet training loop.

NEW in this version:
  • Every data directory is a CLI flag with a sensible Colab default.
  • Startup log prints the exact paths and the file counts found in each.
  • If val pairs aren't found we print WHY (mismatched filenames vs counts).
"""
from __future__ import annotations

import argparse, csv, logging, math, time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.models.voice_filter_model import (
    StudioEnhanceNet, reconstruct_audio, N_FFT, HOP,
)
from src.training.dataset import SpeechEnhancementDataset
from src.training.loss    import SpeechEnhancementLoss


# ── Defaults ────────────────────────────────────────────────────────────────
EPOCHS, BATCH_SIZE, ACCUM_STEPS  = 150, 4, 4
LR, LR_MIN, GRAD_CLIP            = 1e-4, 1e-6, 1.0
EARLY_STOP_PAT, NUM_WORKERS      = 30, 0
DATASET_LEN                      = 2_000
WARMUP_STEPS                     = 200
EMA_DECAY                        = 0.99
LOSS_ALPHA, LOSS_BETA, LOSS_GAMMA = 1.0, 0.3, 0.2

CKPT_DIR = Path("checkpoints")
LOG_PATH = CKPT_DIR / "training_log.csv"

# Data path defaults — match Colab's /content/data_local layout.
# Override with --train-clean, --val-clean, etc. on any other machine.
DEFAULT_TRAIN_CLEAN = "/content/data_local/clean"
DEFAULT_TRAIN_NOISE = "/content/data_local/noise"
DEFAULT_TRAIN_RIR   = "/content/data_local/rir"
DEFAULT_VAL_CLEAN   = "/content/data_local/eval/clean"
DEFAULT_VAL_NOISY   = "/content/data_local/eval/noisy"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train")


# ── STFT ────────────────────────────────────────────────────────────────────
def stft_pair(wav: torch.Tensor):
    win = torch.hann_window(N_FFT, device=wav.device)
    spec = torch.stft(wav, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                      window=win, center=True, return_complex=True)
    return spec.real, spec.imag


# ── EMA ─────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay  = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
                else:
                    self.shadow[k] = v.detach().clone()
    def state_dict(self): return self.shadow


# ── Warmup-cosine LR ────────────────────────────────────────────────────────
def make_lr_lambda(total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(LR_MIN / LR, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


# ── Train / eval epoch ──────────────────────────────────────────────────────
def run_epoch(model, loader, loss_fn, optim, sched, scaler, ema,
              device, train, accum_steps=ACCUM_STEPS):
    model.train() if train else model.eval()
    totals = {"sisnr_db": 0.0, "pcm": 0.0, "mrstft": 0.0, "total": 0.0, "n": 0}
    ctx = torch.enable_grad() if train else torch.no_grad()
    if train: optim.zero_grad(set_to_none=True)

    with ctx:
        for step, batch in enumerate(loader):
            clean = batch["clean_wav"].to(device, non_blocking=True)
            noisy = batch["noisy_wav"].to(device, non_blocking=True)

            with autocast(device_type=device, enabled=(scaler is not None)):
                n_real, n_imag = stft_pair(noisy)
                y_real, y_imag = model(n_real, n_imag)
                est_wav = reconstruct_audio(y_real, y_imag, length=clean.shape[-1])
                loss, parts = loss_fn(est_wav, clean)
                if train: loss = loss / accum_steps

            if train:
                if scaler is not None: scaler.scale(loss).backward()
                else:                  loss.backward()

                if (step + 1) % accum_steps == 0:
                    optim_stepped = False
                    if scaler is not None:
                        scale_before = scaler.get_scale()
                        scaler.unscale_(optim)
                        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                        scaler.step(optim)
                        scaler.update()
                        optim_stepped = scaler.get_scale() >= scale_before
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                        optim.step()
                        optim_stepped = True
                    optim.zero_grad(set_to_none=True)
                    if optim_stepped:
                        sched.step()
                        if ema is not None: ema.update(model)

            for k, v in parts.items():
                if k in totals: totals[k] += float(v)
            totals["n"] += 1

    n = max(totals["n"], 1)
    return {k: v / n for k, v in totals.items() if k != "n"}


# ── STFT round-trip check ───────────────────────────────────────────────────
def _stft_roundtrip_check():
    x = torch.randn(2, 48_000)
    r, i = stft_pair(x)
    y = reconstruct_audio(r, i, length=x.shape[-1])
    err = (x - y).abs().mean().item()
    assert err < 1e-3, f"STFT round-trip broken (err={err:.6f})."
    log.info(f"STFT round-trip OK (mean abs err={err:.2e})")


# ── Path validation — fail loud if val pairs won't match ────────────────────
def _validate_paths(args) -> None:
    """Print exactly what's in each dir AND why val will or won't have pairs."""
    log.info("─── Data paths ───")
    log.info(f"  train clean : {args.train_clean}")
    log.info(f"  train noise : {args.train_noise}")
    log.info(f"  train rir   : {args.train_rir}")
    log.info(f"  val   clean : {args.val_clean}")
    log.info(f"  val   noisy : {args.val_noisy}")

    def count(d):
        p = Path(d)
        if not p.exists(): return 0
        return len(list(p.glob('*.wav')))

    n_tc = count(args.train_clean)
    n_tn = count(args.train_noise)
    n_vc = count(args.val_clean)
    n_vn = count(args.val_noisy)

    log.info(f"  files       : train clean={n_tc}  noise={n_tn}  | "
             f"val clean={n_vc}  noisy={n_vn}")

    if n_tc == 0:
        raise RuntimeError(f"FATAL: no .wav files in train clean dir {args.train_clean}")

    # Count actual matches between val clean and val noisy by FILENAME.
    if n_vc > 0 and n_vn > 0:
        clean_names = {p.name for p in Path(args.val_clean).glob('*.wav')}
        noisy_names = {p.name for p in Path(args.val_noisy).glob('*.wav')}
        matched = clean_names & noisy_names
        log.info(f"  val pairs   : {len(matched)} matched filenames "
                 f"({len(clean_names - noisy_names)} clean orphans, "
                 f"{len(noisy_names - clean_names)} noisy orphans)")
        if not matched:
            sample_c = sorted(list(clean_names))[:3]
            sample_n = sorted(list(noisy_names))[:3]
            log.warning(f"  val sample clean: {sample_c}")
            log.warning(f"  val sample noisy: {sample_n}")
            log.warning("  -> Validation will be a no-op until filenames match.")
    else:
        log.warning("  val dirs missing or empty → validation will be a no-op.")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=EPOCHS)
    p.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--accum-steps", type=int,   default=ACCUM_STEPS)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--dataset-len", type=int,   default=DATASET_LEN)
    p.add_argument("--no-amp",      action="store_true")
    p.add_argument("--no-ema",      action="store_true")

    # ── Data paths (override anywhere) ──────────────────────────────────────
    p.add_argument("--train-clean", default=DEFAULT_TRAIN_CLEAN)
    p.add_argument("--train-noise", default=DEFAULT_TRAIN_NOISE)
    p.add_argument("--train-rir",   default=DEFAULT_TRAIN_RIR)
    p.add_argument("--val-clean",   default=DEFAULT_VAL_CLEAN)
    p.add_argument("--val-noisy",   default=DEFAULT_VAL_NOISY)
    args = p.parse_args()

    _stft_roundtrip_check()
    _validate_paths(args)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Training on {device}")

    train_set = SpeechEnhancementDataset(
        clean_dir = args.train_clean,
        noise_dir = args.train_noise,
        rir_dir   = args.train_rir,
        length    = args.dataset_len,
    )
    val_set = SpeechEnhancementDataset(
        clean_dir = args.val_clean,
        noisy_dir = args.val_noisy,
        eval_mode = True,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True,
                              pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))

    model   = StudioEnhanceNet().to(device)
    loss_fn = SpeechEnhancementLoss(alpha=LOSS_ALPHA, beta=LOSS_BETA,
                                    gamma=LOSS_GAMMA).to(device)
    optim   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    total_steps  = (args.epochs * len(train_loader)) // max(1, args.accum_steps)
    warmup_steps = min(WARMUP_STEPS, max(20, total_steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, make_lr_lambda(total_steps, warmup_steps))

    scaler = GradScaler('cuda') if (device == "cuda" and not args.no_amp) else None
    ema    = None if args.no_ema else EMA(model, EMA_DECAY)

    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    log.info(f"Effective batch  : {args.batch_size} * {args.accum_steps} = "
             f"{args.batch_size * args.accum_steps}")
    log.info(f"AMP={scaler is not None}  EMA={ema is not None} (decay={EMA_DECAY})  "
             f"warmup={warmup_steps}  total_steps={total_steps}")
    log.info(f"Loss weights: alpha={LOSS_ALPHA} (SI-SNR), "
             f"beta={LOSS_BETA} (PCM), gamma={LOSS_GAMMA} (MR-STFT)")

    best_val, bad_epochs = float("inf"), 0
    log_file = open(LOG_PATH, "a", newline="")
    writer = csv.writer(log_file)
    if LOG_PATH.stat().st_size == 0:
        writer.writerow(["epoch", "train_loss", "val_loss",
                         "train_sisnr", "val_sisnr", "lr", "sec"])

    def _val_step():
        if ema is None:
            return run_epoch(model, val_loader, loss_fn, optim, sched,
                             scaler, ema, device, train=False)
        val_model = deepcopy(model)
        val_model.load_state_dict(ema.state_dict())
        val_model.eval()
        return run_epoch(val_model, val_loader, loss_fn, optim, sched,
                         scaler, None, device, train=False)

    for epoch in range(args.epochs):
        t0 = time.time()
        tr = run_epoch(model, train_loader, loss_fn, optim, sched,
                       scaler, ema, device, train=True,
                       accum_steps=args.accum_steps)
        vl = _val_step()
        dt = time.time() - t0

        log.info(f"[{epoch+1:3d}/{args.epochs}] "
                 f"train loss={tr['total']:+.3f} SI-SNR={tr['sisnr_db']:+.2f}dB | "
                 f"val loss={vl['total']:+.3f} SI-SNR={vl['sisnr_db']:+.2f}dB | "
                 f"lr={sched.get_last_lr()[0]:.2e} | {dt:.0f}s")
        writer.writerow([epoch+1, tr["total"], vl["total"],
                         tr["sisnr_db"], vl["sisnr_db"],
                         sched.get_last_lr()[0], int(dt)])
        log_file.flush()

        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "ema":   ema.state_dict() if ema else None},
                   CKPT_DIR / "latest_model.pth")

        best_state = ema.state_dict() if ema is not None else model.state_dict()
        if vl["total"] < best_val:
            best_val, bad_epochs = vl["total"], 0
            torch.save({"model": best_state, "best_val": best_val},
                       CKPT_DIR / "best_model.pth")
            log.info(f"  -> new best {best_val:+.3f}; saved best_model.pth")
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PAT:
                log.info("Early stopping."); break

    log_file.close()
    log.info("Training complete.")


if __name__ == "__main__":
    main()