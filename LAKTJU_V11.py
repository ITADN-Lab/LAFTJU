"""
LAKTJU V11: KF-Enhanced AdamW — Kronecker Factor Directional Enhancement.

V11 核心改进（基于V9/V10失败分析）：
  V9失败原因：SGD-momentum norm(~0.15) 远小于AdamW norm(~43.9)，主路径贡献<1%
  V10失败原因：Norm-matching放大噪声(noKF)或破坏KF量级(with KF)，更差

  V11根本性架构变革：
  - 不再混合两个优化器（彻底放弃homotopy blending架构）
  - 使用单一AdamW作为基础优化器
  - KF提供方向增强：旋转AdamW更新方向以捕获参数间相关性
  - 余弦相似度门控：自适应控制KF修正强度
  - KF warmup：让因子矩阵稳定后再介入
"""
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import math


class LAKTJU_V11(Optimizer):
    r"""
    LAKTJU V11: KF-Enhanced AdamW.

    Architecture:
      1. Compute standard AdamW update: d_adam = m_hat / (sqrt(v_hat) + eps)
      2. Apply KF rotation: d_kf = G_inv @ d_adam_mat @ A_inv
      3. Normalize d_kf to match d_adam's norm (direction-only transfer)
      4. Gate by cosine similarity: effective_alpha = alpha_kf * max(cos_sim, 0)
      5. Blend: d_final = (1 - eff_alpha) * d_adam + eff_alpha * d_kf_normalized
      6. Apply decoupled weight decay + lr scaling

    Why this works:
      - No scale mismatch: both vectors have same norm by construction
      - KF provides directional info only (cross-param correlations AdamW misses)
      - Cosine gate prevents harmful rotations when KF is unreliable
      - Falls back to pure AdamW gracefully (alpha=0 or pre-warmup)
    """

    def __init__(
            self,
            params,
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=5e-4,
            alpha_kf=0.2,
            kf_warmup=500,
            kf_update_interval=20,
            kf_ema=0.95,
            kf_damping=1e-3,
            cos_sim_threshold=0.0,
            grad_clip=0.0,
            warmup=100,
            diag_interval=50,
            alpha_kf_ramp=0,
            kf_residual=False,
            total_steps=0,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, eps=eps,
            weight_decay=weight_decay, alpha_kf=alpha_kf,
            kf_warmup=kf_warmup, grad_clip=grad_clip, warmup=warmup,
        )
        self.kf_update_interval = kf_update_interval
        self.kf_ema = kf_ema
        self.kf_damping = kf_damping
        self.cos_sim_threshold = cos_sim_threshold
        self.diag_interval = diag_interval
        self.alpha_kf_ramp = alpha_kf_ramp
        self.kf_residual = kf_residual
        self.total_steps = total_steps

        # Global state
        self._global_step = 0

        # Kronecker factor storage
        self._kf_A = {}
        self._kf_G = {}
        self._kf_A_inv = {}
        self._kf_G_inv = {}
        self._kf_hooks = []
        self._param_to_module = {}

        # Flag to skip KF accumulation (used during SAM perturbation)
        self._skip_kf = False

        # Diagnostics
        self._diagnostics = {}

        super(LAKTJU_V11, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LAKTJU_V11, self).__setstate__(state)

    # ==================================================================
    # Kronecker Factor Hooks (reused from V9)
    # ==================================================================
    def register_hooks(self, model):
        """Register forward/backward hooks on Conv2d and Linear layers."""
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                for p in module.parameters():
                    self._param_to_module[p.data_ptr()] = module
                h_fwd = module.register_forward_pre_hook(self._forward_hook)
                h_bwd = module.register_full_backward_hook(self._backward_hook)
                self._kf_hooks.extend([h_fwd, h_bwd])

    def disable_kf_hooks(self):
        self._skip_kf = True

    def enable_kf_hooks(self):
        self._skip_kf = False

    def _forward_hook(self, module, input):
        if self._skip_kf or not torch.is_grad_enabled():
            return
        a = input[0].detach()
        if isinstance(module, nn.Conv2d):
            a = self._extract_conv_input(module, a)
        elif isinstance(module, nn.Linear):
            if a.dim() > 2:
                a = a.reshape(-1, a.size(-1))
        if module.bias is not None:
            ones = torch.ones(a.size(0), 1, device=a.device, dtype=a.dtype)
            a = torch.cat([a, ones], dim=1)
        A_batch = a.t() @ a / a.size(0)
        mid = module
        if mid not in self._kf_A:
            self._kf_A[mid] = A_batch
        else:
            self._kf_A[mid].mul_(self.kf_ema).add_(A_batch, alpha=1 - self.kf_ema)

    def _backward_hook(self, module, grad_input, grad_output):
        if self._skip_kf:
            return
        g = grad_output[0].detach()
        if isinstance(module, nn.Conv2d):
            g = g.mean(dim=[2, 3])
        elif isinstance(module, nn.Linear):
            if g.dim() > 2:
                g = g.reshape(-1, g.size(-1))
        G_batch = g.t() @ g / g.size(0)
        mid = module
        if mid not in self._kf_G:
            self._kf_G[mid] = G_batch
        else:
            self._kf_G[mid].mul_(self.kf_ema).add_(G_batch, alpha=1 - self.kf_ema)

    @staticmethod
    def _extract_conv_input(module, a):
        k = module.kernel_size
        s = module.stride
        p = module.padding
        a_unf = torch.nn.functional.unfold(a, kernel_size=k, stride=s, padding=p)
        a_avg = a_unf.mean(dim=2)
        return a_avg

    def _update_kf_inverses(self):
        damping = self.kf_damping
        for module in self._kf_A:
            if module in self._kf_G:
                A = self._kf_A[module]
                G = self._kf_G[module]
                d_a = A.size(0)
                d_g = G.size(0)
                a_damp = max(damping, damping * A.trace().item() / d_a) if d_a > 0 else damping
                g_damp = max(damping, damping * G.trace().item() / d_g) if d_g > 0 else damping
                try:
                    self._kf_A_inv[module] = torch.linalg.inv(
                        A + a_damp * torch.eye(d_a, device=A.device, dtype=A.dtype))
                    self._kf_G_inv[module] = torch.linalg.inv(
                        G + g_damp * torch.eye(d_g, device=G.device, dtype=G.dtype))
                except torch._C._LinAlgError:
                    pass

    def _has_kf(self, p):
        mid = self._param_to_module.get(p.data_ptr())
        return (mid is not None and mid in self._kf_A_inv and mid in self._kf_G_inv)

    def _kf_precondition(self, p, direction):
        """Apply KF preconditioning: G_inv @ direction_matrix @ A_inv."""
        mid = self._param_to_module[p.data_ptr()]
        G_inv = self._kf_G_inv[mid]
        A_inv = self._kf_A_inv[mid]
        d_out = G_inv.size(0)

        if isinstance(mid, nn.Conv2d):
            d_in_kf = A_inv.size(0)
            grad_mat = direction.reshape(d_out, -1)
            if mid.bias is not None and grad_mat.size(1) < d_in_kf:
                grad_mat = torch.cat([grad_mat,
                    torch.zeros(d_out, 1, device=grad_mat.device, dtype=grad_mat.dtype)], dim=1)
        elif isinstance(mid, nn.Linear):
            if mid.bias is not None and direction.dim() == 1 and direction.size(0) == d_out:
                return None  # bias param, skip KF
            grad_mat = direction.reshape(d_out, -1)
            if grad_mat.size(1) < A_inv.size(0):
                grad_mat = torch.cat([grad_mat,
                    torch.zeros(d_out, A_inv.size(0) - grad_mat.size(1),
                                device=grad_mat.device, dtype=grad_mat.dtype)], dim=1)
        else:
            return None

        precond = G_inv @ grad_mat @ A_inv

        orig_numel = direction.numel()
        precond_flat = precond.reshape(-1)[:orig_numel]
        result = precond_flat.reshape(direction.shape)

        return result

    # ==================================================================
    # Main step: KF-Enhanced AdamW
    # ==================================================================
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1

        # Periodically recompute KF inverses (after kf_warmup)
        kf_warmup_steps = self.param_groups[0].get('kf_warmup', 500)
        if (self._global_step % self.kf_update_interval == 0
                and self._global_step > kf_warmup_steps
                and len(self._kf_A) > 0):
            self._update_kf_inverses()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            wd = group['weight_decay']
            alpha_kf = group['alpha_kf']
            kf_warmup = group['kf_warmup']
            warmup_steps = group['warmup']
            grad_clip_val = group['grad_clip']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                t = state['step']

                # Warmup learning rate
                if t <= warmup_steps:
                    current_lr = lr * (t / warmup_steps)
                else:
                    current_lr = lr

                # Optional gradient clipping (per-param)
                if grad_clip_val > 0:
                    gn = grad.norm()
                    if gn > grad_clip_val:
                        grad = grad * (grad_clip_val / (gn + 1e-12))

                # ========================================
                # Standard AdamW moment updates
                # ========================================
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = exp_avg / (1 - beta1 ** t)
                v_hat = exp_avg_sq / (1 - beta2 ** t)

                # AdamW update direction
                d_adam = m_hat / (v_hat.sqrt() + eps)

                # ========================================
                # KF directional enhancement
                # ========================================
                kf_applied = False
                cos_sim_val = 0.0
                eff_alpha = 0.0

                if (alpha_kf > 0
                        and self._global_step > kf_warmup
                        and self._has_kf(p)
                        and p.dim() >= 2):

                    d_kf_raw = self._kf_precondition(p, d_adam)

                    if d_kf_raw is not None:
                        d_adam_norm = d_adam.norm()
                        d_kf_norm = d_kf_raw.norm()

                        if d_kf_norm > 1e-8 and d_adam_norm > 1e-8:
                            # Normalize KF direction to match AdamW magnitude
                            d_kf_normalized = d_kf_raw * (d_adam_norm / d_kf_norm)

                            # Cosine similarity gate
                            cos_sim_val = (d_adam * d_kf_raw).sum() / (d_adam_norm * d_kf_norm)

                            if cos_sim_val > self.cos_sim_threshold:
                                # Adaptive alpha: scale by agreement
                                eff_alpha = alpha_kf * max(cos_sim_val.item(), 0.0)

                                d_adam = (1 - eff_alpha) * d_adam + eff_alpha * d_kf_normalized
                                kf_applied = True

                # ========================================
                # Decoupled weight decay (standard AdamW)
                # ========================================
                if wd != 0:
                    p.mul_(1.0 - current_lr * wd)

                # Apply update
                p.add_(d_adam, alpha=-current_lr)

                # ========================================
                # Diagnostics
                # ========================================
                if self.diag_interval > 0 and self._global_step % self.diag_interval == 0:
                    if not self._diagnostics:
                        self._diagnostics = {
                            'adam_update_norm': 0.0,
                            'kf_applied_count': 0,
                            'kf_skipped_count': 0,
                            'total_param_count': 0,
                            'avg_cos_sim': 0.0,
                            'avg_eff_alpha': 0.0,
                            'cos_sim_sum': 0.0,
                            'eff_alpha_sum': 0.0,
                        }
                    self._diagnostics['adam_update_norm'] += d_adam.norm().item()
                    self._diagnostics['total_param_count'] += 1
                    if kf_applied:
                        self._diagnostics['kf_applied_count'] += 1
                        self._diagnostics['cos_sim_sum'] += cos_sim_val.item() if isinstance(cos_sim_val, torch.Tensor) else cos_sim_val
                        self._diagnostics['eff_alpha_sum'] += eff_alpha
                    elif self._has_kf(p) and p.dim() >= 2:
                        self._diagnostics['kf_skipped_count'] += 1

        return loss

    def get_diagnostics(self):
        """Return and clear diagnostics dict."""
        d = self._diagnostics.copy()
        if d.get('kf_applied_count', 0) > 0:
            d['avg_cos_sim'] = d['cos_sim_sum'] / d['kf_applied_count']
            d['avg_eff_alpha'] = d['eff_alpha_sum'] / d['kf_applied_count']
        self._diagnostics = {}
        return d
