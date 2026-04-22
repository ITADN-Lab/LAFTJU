import torch
from torch.optim import AdamW
import math


@torch.no_grad()
def _ns_ortho_bf16(G, ns_steps=1):
    """Newton-Schulz quintic orthogonalization in bfloat16."""
    m, n = G.shape
    transposed = m > n
    if transposed:
        G = G.T
        m, n = n, m

    norm = G.norm()
    if norm < 1e-12:
        return G.T if transposed else G
    X = G / norm

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(ns_steps):
        A = X @ X.T
        AX = A @ X
        X = a * X + b * AX + c * (A @ AX)

    return X.T if transposed else X


class LAKTJU_NS(AdamW):
    r"""LAKTJU-NS v8: AdamW + periodic NS momentum orthogonalization.

    Inherits from PyTorch's fused C++ AdamW for maximum speed.
    Adds periodic Newton-Schulz orthogonalization of the momentum
    buffer (exp_avg) every ns_interval steps.

    Non-NS steps are IDENTICAL to PyTorch AdamW — zero overhead.
    NS steps add a brief orthogonalization pass after the AdamW update.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, ns_interval=500, ns_steps=1, min_ndim=2,
                 ns_min_dim=1, ns_max_dim=256, grad_centralization=True,
                 ns_skip_params=None,
                 # Unused but kept for API compatibility
                 warmup=0, total_steps=None, homotopy_speed=1.0):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay)
        self.ns_interval = ns_interval
        self.ns_steps = ns_steps
        self.ns_min_ndim = min_ndim
        self.ns_min_dim = ns_min_dim
        self.ns_max_dim = ns_max_dim
        self.grad_centralization = grad_centralization
        # Set of parameter ids to skip NS orthogonalization
        self._ns_skip_ids = set(id(p) for p in (ns_skip_params or []))
        self._ns_step_counter = 0

    @torch.no_grad()
    def step(self, closure=None):
        # Apply gradient centralization before AdamW step
        if self.grad_centralization:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None and p.grad.ndim >= 4:
                        p.grad.sub_(p.grad.mean(
                            dim=tuple(range(1, p.grad.ndim)), keepdim=True))

        # Call PyTorch's fused C++ AdamW
        loss = super().step(closure)

        self._ns_step_counter += 1

        # Periodically orthogonalize momentum buffers
        if self._ns_step_counter % self.ns_interval == 0:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None or p.ndim < self.ns_min_ndim:
                        continue
                    if id(p) in self._ns_skip_ids:
                        continue
                    state = self.state[p]
                    if 'exp_avg' not in state:
                        continue
                    m1 = state['exp_avg']
                    shape = m1.shape
                    rows = shape[0]
                    cols = m1.numel() // rows
                    mdim = min(rows, cols)
                    if not (self.ns_min_dim <= mdim <= self.ns_max_dim):
                        continue

                    G = m1.reshape(rows, cols).to(torch.bfloat16)
                    U = _ns_ortho_bf16(G, ns_steps=self.ns_steps)
                    m1_norm = m1.norm()
                    u_norm = U.norm()
                    if u_norm > 1e-12:
                        m1.copy_(U.to(m1.dtype).reshape(shape).mul_(
                            m1_norm / u_norm))

        return loss
