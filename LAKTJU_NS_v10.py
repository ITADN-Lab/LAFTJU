"""
LAKTJU-NS v10: Gate-wise NS orthogonalization for LSTM + L2 regularization mode.

Key innovations over v8:
1. Gate-wise NS: LSTM weight matrices (4H, H) are split into 4 gate blocks (H, H)
   and each gate is orthogonalized independently. Square (H,H) matrices are ideal
   for NS — they converge in 1-2 steps to near-perfect orthogonality.
2. L2 regularization mode: weight_decay is folded into the gradient (like Adam),
   not applied as decoupled weight decay (like AdamW). This is known to be better
   for LSTM recurrent weights (Merity et al. 2018).
"""

import torch
from torch.optim import Optimizer
import math


@torch.no_grad()
def _ns_ortho_bf16(G, ns_steps=2):
    """Newton-Schulz quintic orthogonalization in bfloat16."""
    m, n = G.shape
    transposed = m > n
    if transposed:
        G = G.T
        m, n = n, m

    norm = G.norm()
    if norm < 1e-12:
        return G.T if transposed else G
    X = G.to(torch.bfloat16) / norm.to(torch.bfloat16)

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(ns_steps):
        A = X @ X.T
        AX = A @ X
        X = a * X + b * AX + c * (A @ AX)

    return (X.T if transposed else X).to(G.dtype)


@torch.no_grad()
def _ns_gatewise(m1, nhid, ns_steps=2):
    """
    Gate-wise NS for LSTM weight matrices of shape (4*nhid, input_dim).
    Splits into 4 gate blocks of (nhid, input_dim) and orthogonalizes each.
    When nhid == input_dim, each block is a square matrix — ideal for NS.
    """
    rows, cols = m1.shape  # (4*nhid, input_dim)
    result = torch.empty_like(m1)
    for gate_idx in range(4):
        start = gate_idx * nhid
        end = start + nhid
        block = m1[start:end, :]  # (nhid, input_dim)
        block_norm = block.norm()
        if block_norm < 1e-12:
            result[start:end, :] = block
            continue
        U = _ns_ortho_bf16(block, ns_steps=ns_steps)
        u_norm = U.norm()
        if u_norm > 1e-12:
            result[start:end, :] = U.mul_(block_norm / u_norm)
        else:
            result[start:end, :] = block
    return result


class LAKTJU_NS_v10(Optimizer):
    """
    LAKTJU-NS v10: Adam with L2 regularization + gate-wise NS for LSTM.

    Uses Adam (not AdamW) as the base — weight_decay is folded into the
    gradient as L2 regularization, which is known to be better for LSTM.

    NS orthogonalization is applied gate-wise to LSTM weight matrices,
    treating each (nhid, input_dim) gate block independently.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1.2e-6,
                 ns_interval=50, ns_steps=2,
                 nhid=650,
                 grad_centralization=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.ns_interval = ns_interval
        self.ns_steps = ns_steps
        self.nhid = nhid
        self.grad_centralization = grad_centralization
        self._step_count = 0

    def _is_lstm_gate_weight(self, p):
        """Detect LSTM gate weight: shape (4*nhid, any)."""
        return p.ndim == 2 and p.shape[0] == 4 * self.nhid

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        do_ns = (self._step_count % self.ns_interval == 0)

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.clone()

                # L2 regularization: fold weight_decay into gradient (Adam-style)
                if wd != 0:
                    grad = grad.add(p.data, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                t = state['step']
                m1 = state['exp_avg']
                m2 = state['exp_avg_sq']

                # Adam moment updates
                m1.mul_(beta1).add_(grad, alpha=1 - beta1)
                m2.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t
                step_size = lr / bc1
                denom = (m2.sqrt() / math.sqrt(bc2)).add_(eps)

                # Parameter update
                p.data.addcdiv_(m1, denom, value=-step_size)

                # Gate-wise NS orthogonalization of momentum buffer
                if do_ns and self._is_lstm_gate_weight(p):
                    m1.copy_(_ns_gatewise(m1, self.nhid, self.ns_steps))

        return loss
