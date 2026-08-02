"""Combined loss: -alpha*SI-SNR + beta*PCM + gamma*MR-STFT.
NEW DEFAULTS: alpha=1.0 (was 0.05) so SI-SNR drives optimisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def si_snr(est: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    est    = est    - est.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    dot      = (est * target).sum(dim=-1, keepdim=True)
    t_energy = (target * target).sum(dim=-1, keepdim=True) + eps
    s_target = dot / t_energy * target
    e_noise  = est - s_target
    s_pow = (s_target * s_target).sum(dim=-1) + eps
    n_pow = (e_noise  * e_noise ).sum(dim=-1) + eps
    return (10.0 * torch.log10(s_pow / n_pow)).clamp(min=-30.0, max=35.0)


class STFTLoss(nn.Module):
    def __init__(self, n_fft, hop, win):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self._cache = {}

    def _window(self, device):
        key = str(device)
        if key not in self._cache:
            self._cache[key] = torch.hann_window(self.win, device=device)
        return self._cache[key]

    def stft_mag(self, x):
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                          win_length=self.win, window=self._window(x.device),
                          center=True, return_complex=True)
        return torch.abs(spec)

    def forward(self, est, ref):
        e = self.stft_mag(est) + 1e-7
        r = self.stft_mag(ref) + 1e-7
        sc = torch.norm(r - e, p='fro') / (torch.norm(r, p='fro') + 1e-7)
        return sc + F.l1_loss(torch.log(e), torch.log(r))


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = nn.ModuleList([
            STFTLoss(512,  50,  240),
            STFTLoss(1024, 120, 600),
            STFTLoss(2048, 240, 1200),
        ])

    def forward(self, est, ref):
        return sum(l(est, ref) for l in self.losses) / len(self.losses)


class PCMSpectralLoss(nn.Module):
    def __init__(self, n_fft: int = 512, hop: int = 128, p: float = 0.3):
        super().__init__()
        self.n_fft, self.hop, self.p = n_fft, hop, p
        self._cache = {}

    def _window(self, device):
        if str(device) not in self._cache:
            self._cache[str(device)] = torch.hann_window(self.n_fft, device=device)
        return self._cache[str(device)]

    def forward(self, est, ref):
        spec_e = torch.stft(est, self.n_fft, self.hop, self.n_fft,
                            self._window(est.device), center=True, return_complex=True)
        spec_r = torch.stft(ref, self.n_fft, self.hop, self.n_fft,
                            self._window(ref.device), center=True, return_complex=True)
        mag_e = (spec_e.abs() + 1e-7).pow(self.p)
        mag_r = (spec_r.abs() + 1e-7).pow(self.p)
        ce = spec_e * (spec_e.abs() + 1e-7).pow(self.p - 1)
        cr = spec_r * (spec_r.abs() + 1e-7).pow(self.p - 1)
        return F.l1_loss(mag_e, mag_r) + F.l1_loss(ce.real, cr.real) \
                                       + F.l1_loss(ce.imag, cr.imag)


class SpeechEnhancementLoss(nn.Module):
    """
    L = -alpha*SI-SNR + beta*PCM + gamma*MR-STFT

    DEFAULTS CHANGED:
        alpha 0.05 -> 1.00   (SI-SNR now dominates)
        beta  0.50 -> 0.30
        gamma 0.30 -> 0.20

    Rationale: SI-SNR is the metric we actually care about, and the
    spectral losses can be locally minimised by phase/alignment patterns
    that destroy waveform quality. We keep them as auxiliary regularisers.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.3, gamma: float = 0.2):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.pcm    = PCMSpectralLoss()
        self.mrstft = MultiResolutionSTFTLoss()

    def forward(self, est_wav, ref_wav):
        n = min(est_wav.shape[-1], ref_wav.shape[-1])
        est_wav, ref_wav = est_wav[..., :n], ref_wav[..., :n]
        sisdr  = si_snr(est_wav, ref_wav).mean()
        pcm    = self.pcm(est_wav, ref_wav)
        mrstft = self.mrstft(est_wav, ref_wav)
        total  = -self.alpha * sisdr + self.beta * pcm + self.gamma * mrstft
        return total, {'sisnr_db': sisdr.item(), 'pcm': pcm.item(),
                       'mrstft':   mrstft.item(), 'total': total.item()}