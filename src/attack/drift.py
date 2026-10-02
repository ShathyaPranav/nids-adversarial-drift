"""Drift detector under attack, plus the differentiable proxies the attacker optimizes.

Detector (not differentiable, used only for evaluation): two-sample Kolmogorov-Smirnov tests
between a clean reference window and the incoming window, one per input feature and one on
the classifier's confidence (max softmax). Bonferroni-corrected; any rejection flags drift.

Proxies (differentiable stand-ins for that detector):
  marginal  per-column quantile matching (1-D Wasserstein per column), which mirrors the
            per-feature structure of the KS tests
  mmd       multi-scale RBF MMD^2 over the joint representation (Xu et al., 2022)
"""
import numpy as np
import torch
from scipy.stats import ks_2samp


class KSDriftDetector:
    def __init__(self, ref_X: np.ndarray, ref_conf: np.ndarray, alpha: float = 0.05):
        self.ref = np.column_stack([ref_X, ref_conf])
        self.alpha = alpha

    def detect(self, X: np.ndarray, conf: np.ndarray) -> dict:
        window = np.column_stack([X, conf])
        n_tests = window.shape[1]
        pvals = np.array([ks_2samp(self.ref[:, j], window[:, j]).pvalue for j in range(n_tests)])
        rejected = pvals < self.alpha / n_tests
        return {
            "drift": bool(rejected.any()),
            "n_rejected_features": int(rejected[:-1].sum()),
            "confidence_rejected": bool(rejected[-1]),
        }


def representation(z: torch.Tensor, logits: torch.Tensor, kind: str) -> torch.Tensor:
    conf = torch.softmax(logits, dim=1).max(dim=1).values.unsqueeze(1)
    if kind == "features":
        return z
    if kind == "confidence":
        return conf
    if kind == "both":
        return torch.cat([z, conf], dim=1)
    raise ValueError(f"unknown representation: {kind}")


def marginal_proxy(window: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    levels = (torch.arange(window.size(0), device=window.device) + 0.5) / window.size(0)
    ref_q = torch.quantile(ref, levels, dim=0)
    return (window.sort(dim=0).values - ref_q).abs().mean()


def mmd_proxy(window: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    xy = torch.cat([window, ref], dim=0)
    d2 = torch.cdist(xy, xy).pow(2)
    base = d2.detach().median().clamp_min(1e-6)
    k = sum(torch.exp(-d2 / (s * base)) for s in (0.25, 1.0, 4.0))
    n = window.size(0)
    return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()


PROXIES = {"marginal": marginal_proxy, "mmd": mmd_proxy}
