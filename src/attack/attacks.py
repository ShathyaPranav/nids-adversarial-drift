"""Evasion attacks on a window of flows. Only the rows in `adv_mask` are perturbed.

  pgd        classifier evasion only: minimize CE toward the benign class
  decoupled  two stages (Kuppa & Le-Khac-style baseline): classifier evasion for the first
             half of the step budget, then drift reduction alone for the second half
  joint      one objective optimized throughout: CE + lam * drift proxy

All modes share the same step budget and the same constraint projection.
"""
import torch
import torch.nn.functional as F

from .drift import PROXIES, representation

MODES = ("pgd", "decoupled", "joint")


def run_attack(
    model,
    constraints,
    z_window: torch.Tensor,
    adv_mask: torch.Tensor,
    benign_idx: int,
    ref_z: torch.Tensor,
    mode: str,
    eps: float,
    step_size: float,
    steps: int,
    lam: float,
    proxy: str,
    representation_kind: str,
) -> torch.Tensor:
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    proxy_fn = PROXIES[proxy]
    model.eval()

    with torch.no_grad():
        ref_repr = representation(ref_z, model(ref_z), representation_kind)

    target = torch.full((int(adv_mask.sum()),), benign_idx, device=z_window.device)
    row_mask = adv_mask.unsqueeze(1)
    z = z_window.clone()

    for t in range(steps):
        z.requires_grad_(True)
        logits = model(z)
        ce = F.cross_entropy(logits[adv_mask], target)
        drift = proxy_fn(representation(z, logits, representation_kind), ref_repr)
        if mode == "pgd":
            loss = ce
        elif mode == "joint":
            loss = ce + lam * drift
        else:
            loss = ce if t < steps // 2 else drift
        (grad,) = torch.autograd.grad(loss, z)
        with torch.no_grad():
            z_prev = z.detach()
            z = z_prev - step_size * grad.sign() * row_mask
            z = torch.where(row_mask, constraints.project(z, z_window, eps, z_prev), z_window)
    return z.detach()
