"""Problem-space constraints for adversarial flows (after Pierazzi et al., 2020).

The classifier sees z = (signed_log1p(raw) - mean) / scale. Constraints are stated on raw
feature values and enforced on z:
  - immutable features are never changed
  - each mutable feature moves at most `eps` (standardized units) from its original value
  - mutable features stay non-negative (they are counts, sizes, durations, rates); the few
    clean flows that already hold a negative value (CICFlowMeter's header-length overflow)
    may not go lower than they started
  - Min <= Mean <= Max holds within each (Min, Mean, Max) feature family

Not enforced: exact arithmetic identities between features (e.g. total = mean * count) and
integer-valued counts. A flow that passes these checks is plausible, not guaranteed realizable.
"""
import torch

ORDER_TOL = 1e-3


class FeatureConstraints:
    def __init__(self, feature_cols: list[str], mean, scale, cfg: dict, device):
        self.feature_cols = feature_cols
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        self.scale = torch.as_tensor(scale, dtype=torch.float32, device=device)

        exact = set(cfg.get("immutable_exact", []))
        contains = cfg.get("immutable_contains", [])
        unknown = exact - set(feature_cols)
        if unknown:
            raise ValueError(f"immutable_exact names not in feature list: {sorted(unknown)}")
        immutable = [c in exact or any(s in c for s in contains) for c in feature_cols]
        self.mutable = ~torch.tensor(immutable, device=device)

        # z-value that corresponds to raw value 0 (signed_log1p(0) == 0)
        self.z_zero = -self.mean / self.scale

        self.triples = []
        index = {c: i for i, c in enumerate(feature_cols)}
        for c, i in index.items():
            if c.endswith(" Min"):
                stem = c[: -len(" Min")]
                mid, hi = index.get(f"{stem} Mean"), index.get(f"{stem} Max")
                if mid is not None and hi is not None and all(self.mutable[j] for j in (i, mid, hi)):
                    self.triples.append((i, mid, hi))

    def mutable_names(self) -> list[str]:
        return [c for c, m in zip(self.feature_cols, self.mutable.tolist()) if m]

    def _u(self, z: torch.Tensor, i: int) -> torch.Tensor:
        return z[:, i] * self.scale[i] + self.mean[i]

    def _order_broken(self, z: torch.Tensor, triple) -> torch.Tensor:
        lo, mid, hi = triple
        u_lo, u_mid, u_hi = self._u(z, lo), self._u(z, mid), self._u(z, hi)
        return (u_lo > u_mid + ORDER_TOL) | (u_mid > u_hi + ORDER_TOL)

    @torch.no_grad()
    def project(self, z_adv: torch.Tensor, z_orig: torch.Tensor, eps: float, z_prev: torch.Tensor) -> torch.Tensor:
        """Map z_adv to a valid flow. `z_prev` is the last valid iterate (z_orig at step 0)."""
        delta = (z_adv - z_orig).clamp(-eps, eps) * self.mutable
        z = z_orig + delta
        floor = torch.minimum(self.z_zero, z_orig)
        z = torch.where(self.mutable & (z < floor), floor, z)
        # a family whose ordering broke keeps its previous valid values, which stay within eps
        for triple in self.triples:
            broken = self._order_broken(z, triple)
            for i in triple:
                z[:, i] = torch.where(broken, z_prev[:, i], z[:, i])
        return z

    @torch.no_grad()
    def violations(self, z_adv: torch.Tensor, z_orig: torch.Tensor) -> torch.Tensor:
        """Per-flow bool: True if the flow breaks any constraint."""
        immutable_changed = ((z_adv != z_orig) & ~self.mutable).any(dim=1)
        u = z_adv * self.scale + self.mean
        u_floor = torch.minimum(torch.zeros_like(u), z_orig * self.scale + self.mean)
        bad = immutable_changed | ((u < u_floor - ORDER_TOL) & self.mutable).any(dim=1)
        for triple in self.triples:
            bad |= self._order_broken(z_adv, triple)
        return bad
