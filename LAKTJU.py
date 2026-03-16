import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import math


class LAKTJU(Optimizer):
    r"""
    LAKTJU: Layer-wise Adaptive Kronecker-factored Trajectory Unified optimizer.

    TJU-V6 optimizer that extends ATJU with:
      1) Layer-wise adaptive QGS coupling (gradient contribution ratio)
      2) Kronecker-factored PTC (replacing diagonal Hessian approximation)
      3) Curvature-aware homotopy scheduler (replacing fixed sigmoid)
      4) QGS loss amplification (paper Eq 6)
    """

    def __init__(
            self,
            params,
            tju_lr=0.01,
            a_lr=0.001,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=1e-4,
            c_base=1.0,
            kappa=5.0,
            homotopy_sharpness=10.0,
            homotopy_speed=1.0,
            warmup=20,
            total_steps=0,
            kf_update_interval=20,
            kf_ema=0.95,
            kf_damping=1e-3,
    ):
        defaults = dict(
            tju_lr=tju_lr, a_lr=a_lr, beta1=beta1, beta2=beta2, eps=eps,
            weight_decay=weight_decay, c_base=c_base, kappa=kappa,
            homotopy_sharpness=homotopy_sharpness, homotopy_speed=homotopy_speed,
            warmup=warmup, total_steps=total_steps,
            kf_update_interval=kf_update_interval, kf_ema=kf_ema,
            kf_damping=kf_damping,
        )
        self.total_steps = total_steps
        self.c_base = c_base
        self.kappa = kappa
        self.homotopy_sharpness = homotopy_sharpness
        self.homotopy_speed = homotopy_speed
        self.kf_update_interval = kf_update_interval
        self.kf_ema = kf_ema
        self.kf_damping = kf_damping

        # Global state for curvature-aware homotopy
        self._global_step = 0
        self._gamma_0 = None
        self._prev_gamma = None
        self._gamma_ema = None
        self._gamma_ema_beta = 0.95

        # Kronecker factor storage: keyed by module
        self._kf_A = {}   # input covariance factors
        self._kf_G = {}   # gradient covariance factors
        self._kf_A_inv = {}
        self._kf_G_inv = {}
        self._kf_hooks = []
        self._param_to_module = {}  # maps param data_ptr -> module

        # QGS loss amplification
        self._current_loss = None

        super(LAKTJU, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LAKTJU, self).__setstate__(state)

    def set_loss(self, loss_val):
        """Set current loss value for QGS loss amplification (paper Eq 6)."""
        self._current_loss = loss_val

    # ==================================================================
    # Kronecker Factor Hooks
    # ==================================================================
    def register_hooks(self, model):
        """Register forward/backward hooks on Conv2d and Linear layers for KF accumulation."""
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Map parameter data_ptrs to this module
                for p in module.parameters():
                    self._param_to_module[p.data_ptr()] = module

                h_fwd = module.register_forward_pre_hook(self._forward_hook)
                h_bwd = module.register_full_backward_hook(self._backward_hook)
                self._kf_hooks.extend([h_fwd, h_bwd])

    def _forward_hook(self, module, input):
        """Capture input activations -> compute A_l = E[aa^T] (spatial-averaged for conv)."""
        if not torch.is_grad_enabled():
            return
        a = input[0].detach()
        # For Conv2d: spatial-average the unfolded input
        if isinstance(module, nn.Conv2d):
            # a: (B, C_in, H, W) -> unfold -> (B, C_in*k*k, L) -> average over spatial
            a = self._extract_conv_input(module, a)
        elif isinstance(module, nn.Linear):
            # a: (B, in_features) or (B, ..., in_features)
            if a.dim() > 2:
                a = a.reshape(-1, a.size(-1))
        # a: (B, d_in)  -- add bias dimension
        if module.bias is not None:
            ones = torch.ones(a.size(0), 1, device=a.device, dtype=a.dtype)
            a = torch.cat([a, ones], dim=1)

        # A = (1/B) * a^T @ a  -> (d_in, d_in)
        A_batch = a.t() @ a / a.size(0)

        mid = module
        if mid not in self._kf_A:
            self._kf_A[mid] = A_batch
        else:
            self._kf_A[mid].mul_(self.kf_ema).add_(A_batch, alpha=1 - self.kf_ema)

    def _backward_hook(self, module, grad_input, grad_output):
        """Capture output gradients -> compute G_l = E[delta*delta^T] (spatial-averaged for conv)."""
        g = grad_output[0].detach()
        if isinstance(module, nn.Conv2d):
            # g: (B, C_out, H, W) -> (B, C_out) by averaging over spatial dims
            g = g.mean(dim=[2, 3])
        elif isinstance(module, nn.Linear):
            if g.dim() > 2:
                g = g.reshape(-1, g.size(-1))
        # G = (1/B) * g^T @ g -> (d_out, d_out)
        G_batch = g.t() @ g / g.size(0)

        mid = module
        if mid not in self._kf_G:
            self._kf_G[mid] = G_batch
        else:
            self._kf_G[mid].mul_(self.kf_ema).add_(G_batch, alpha=1 - self.kf_ema)

    @staticmethod
    def _extract_conv_input(module, a):
        """Unfold conv input and spatial-average to get (B, C_in*k*k) matrix."""
        # a: (B, C_in, H, W)
        k = module.kernel_size
        s = module.stride
        p = module.padding
        # Unfold: (B, C_in*k_h*k_w, L) where L = output spatial locations
        a_unf = torch.nn.functional.unfold(a, kernel_size=k, stride=s, padding=p)
        # Average over spatial locations: (B, C_in*k_h*k_w)
        a_avg = a_unf.mean(dim=2)
        return a_avg

    def _update_kf_inverses(self):
        """Recompute damped inverses for all tracked modules."""
        damping = self.kf_damping
        for module in self._kf_A:
            if module in self._kf_G:
                A = self._kf_A[module]
                G = self._kf_G[module]
                d_a = A.size(0)
                d_g = G.size(0)
                # Trace-scaled damping: max(damping, damping * trace/dim) for robustness
                a_damp = max(damping, damping * A.trace().item() / d_a) if d_a > 0 else damping
                g_damp = max(damping, damping * G.trace().item() / d_g) if d_g > 0 else damping
                try:
                    self._kf_A_inv[module] = torch.linalg.inv(
                        A + a_damp * torch.eye(d_a, device=A.device, dtype=A.dtype))
                    self._kf_G_inv[module] = torch.linalg.inv(
                        G + g_damp * torch.eye(d_g, device=G.device, dtype=G.dtype))
                except torch._C._LinAlgError:
                    # Fallback: skip this module's KF this round
                    pass

    def _has_kf(self, p):
        """Check if parameter has Kronecker factors available."""
        mid = self._param_to_module.get(p.data_ptr())
        return (mid is not None and mid in self._kf_A_inv and mid in self._kf_G_inv)

    def _kf_precondition(self, p, grad_ema, c_l):
        """Apply Kronecker-factored preconditioning: c_l * G_inv @ grad_matrix @ A_inv."""
        mid = self._param_to_module[p.data_ptr()]
        G_inv = self._kf_G_inv[mid]
        A_inv = self._kf_A_inv[mid]

        d_out = G_inv.size(0)
        # Reshape gradient EMA to matrix form
        if isinstance(mid, nn.Conv2d):
            # weight shape: (C_out, C_in, k_h, k_w) -> (C_out, C_in*k_h*k_w)
            d_in_kf = A_inv.size(0)
            if mid.bias is not None:
                # grad_ema is for weight only; bias handled separately
                grad_mat = grad_ema.reshape(d_out, -1)
                # Pad with zeros for bias column if A_inv includes bias dim
                if grad_mat.size(1) < d_in_kf:
                    grad_mat = torch.cat([grad_mat,
                        torch.zeros(d_out, 1, device=grad_mat.device, dtype=grad_mat.dtype)], dim=1)
            else:
                grad_mat = grad_ema.reshape(d_out, -1)
        elif isinstance(mid, nn.Linear):
            if mid.bias is not None and grad_ema.dim() == 1 and grad_ema.size(0) == d_out:
                # This is the bias parameter - skip KF, return diagonal fallback
                return None
            grad_mat = grad_ema.reshape(d_out, -1)
            if grad_mat.size(1) < A_inv.size(0):
                grad_mat = torch.cat([grad_mat,
                    torch.zeros(d_out, A_inv.size(0) - grad_mat.size(1),
                                device=grad_mat.device, dtype=grad_mat.dtype)], dim=1)
        else:
            return None

        # KF preconditioning: G_inv @ grad_mat @ A_inv
        precond = c_l * (G_inv @ grad_mat @ A_inv)

        # Remove bias column if we padded it
        orig_numel = grad_ema.numel()
        precond_flat = precond.reshape(-1)[:orig_numel]
        result = precond_flat.reshape(grad_ema.shape)

        # Clip KF update to prevent explosion: scale down if norm is too large
        grad_norm = grad_ema.norm()
        result_norm = result.norm()
        if result_norm > 3.0 * grad_norm + 1e-8:
            result = result * (3.0 * grad_norm / (result_norm + 1e-8))

        return result

    # ==================================================================
    # Homotopy
    # ==================================================================
    def _compute_spectral_gap(self, param_states):
        """Compute aggregate spectral gap from approximate Hessian diagonal."""
        gaps = []
        for state in param_states:
            if 'approx_hessian' in state and state['approx_hessian'].numel() > 1:
                h = state['approx_hessian'].abs()
                h_nonzero = h[h > 1e-12]
                if h_nonzero.numel() > 0:
                    gap = h_nonzero.max().item() / (h_nonzero.min().item() + 1e-12)
                    gaps.append(min(gap, 1e6))
        if len(gaps) == 0:
            return 1.0
        return sum(gaps) / len(gaps)

    def _compute_homotopy_s(self, gamma_prev):
        """Curvature-aware homotopy transition variable using tanh."""
        progress = self._global_step / max(self.total_steps, 1)
        s_progress = math.tanh(progress * self.homotopy_speed)

        if self._gamma_0 is None or self._gamma_0 < 1e-12:
            return s_progress

        ratio = self._gamma_0 / (gamma_prev + 1e-12)
        s_curvature = 1.0 / (1.0 + math.exp(-self.homotopy_sharpness * (ratio - 1.0)))

        return max(s_progress, s_curvature)

    # ==================================================================
    # Main step
    # ==================================================================
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1

        # Periodically recompute KF inverses (only after warmup)
        if (self._global_step % self.kf_update_interval == 0
                and self._global_step > 100
                and len(self._kf_A) > 0):
            self._update_kf_inverses()

        for group in self.param_groups:
            tju_lr = group['tju_lr']
            a_lr = group['a_lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            wd = group['weight_decay']
            warmup_steps = group['warmup']
            c_base = group['c_base']
            kappa = group['kappa']

            # Collect layer-wise gradient norms for coupling ratio
            layer_info = []
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                gn = g.norm(2).item()
                np_ = p.numel()
                layer_info.append((p, g, gn, np_))

            if len(layer_info) == 0:
                continue

            # Normalized gradient norms for layer-wise coupling
            normalized_norms = []
            for (p, g, gn, np_) in layer_info:
                normalized_norms.append(gn / (math.sqrt(np_) + 1e-12))
            avg_norm = sum(normalized_norms) / len(normalized_norms) if normalized_norms else 1e-12

            # Compute homotopy s from spectral gap
            param_states = [self.state[p] for (p, _, _, _) in layer_info
                           if p in self.state and len(self.state[p]) > 0]
            current_gamma = self._compute_spectral_gap(param_states)

            if self._gamma_ema is None:
                self._gamma_ema = current_gamma
            else:
                self._gamma_ema = (self._gamma_ema_beta * self._gamma_ema
                                   + (1 - self._gamma_ema_beta) * current_gamma)

            if self._gamma_0 is None:
                progress = self._global_step / max(self.total_steps, 1)
                s = math.tanh(progress * self.homotopy_speed)
            elif self._gamma_ema is not None:
                s = self._compute_homotopy_s(self._gamma_ema)
            else:
                s = 0.0

            self._prev_gamma = current_gamma
            if (self._gamma_0 is None and self._global_step >= warmup_steps
                    and self._gamma_ema is not None):
                self._gamma_0 = self._gamma_ema

            # Per-parameter update
            for idx, (p, grad, grad_norm, n_params) in enumerate(layer_info):
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg_grad'] = torch.zeros_like(p)
                    state['approx_hessian'] = torch.zeros_like(p)
                    state['prev_update'] = torch.zeros_like(p)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                step_count = state['step']

                # Warmup learning rate
                if step_count <= warmup_steps:
                    lr_scale = step_count / warmup_steps
                    current_tju_lr = tju_lr * lr_scale
                    current_a_lr = a_lr * lr_scale
                else:
                    current_tju_lr = tju_lr
                    current_a_lr = a_lr

                # Layer-wise Adaptive QGS Coupling
                rho_l = normalized_norms[idx] / (avg_norm + 1e-12)
                sigma_rho = 2.0 / (1.0 + math.exp(-kappa * (rho_l - 1.0)))

                # QGS loss amplification (paper Eq 6), with conservative cap
                if self._current_loss is not None:
                    loss_factor = min(1.0 + 0.1 * self._current_loss, 1.5)
                    c_l = c_base * loss_factor * sigma_rho
                else:
                    c_l = c_base * sigma_rho

                # ========================================
                # TJU update: KF-preconditioned or diagonal fallback
                # ========================================
                exp_avg_grad = state['exp_avg_grad']
                approx_hessian = state['approx_hessian']
                prev_update = state['prev_update']

                grad_tju = grad.clone()
                bias_corr = 1 - (beta1 ** step_count)
                effective_alpha = (1 - beta1) / bias_corr

                grad_diff = grad_tju - exp_avg_grad
                exp_avg_grad.add_(grad_diff, alpha=effective_alpha)

                # Try KF preconditioning for Conv2d/Linear weights
                update_tju = None
                if self._has_kf(p) and p.dim() >= 2:
                    update_tju = self._kf_precondition(p, exp_avg_grad, c_l)

                # Diagonal fallback for 1D params (BN, biases) or if KF not available
                if update_tju is None:
                    prev_norm = prev_update.norm(p=4).add_(eps)
                    prev_update_normalized = prev_update / prev_norm
                    prev_update_sq = prev_update_normalized * prev_update_normalized

                    delta_term = (
                        (grad_diff / prev_norm * prev_update_normalized).sum()
                        * (-effective_alpha)
                        - (approx_hessian * prev_update_sq).sum()
                    )
                    approx_hessian.addcmul_(prev_update_sq, delta_term)

                    rebound_thresh = 0.01
                    denom_h = approx_hessian.abs().clamp_(min=rebound_thresh)
                    update_tju = c_l * exp_avg_grad / denom_h

                # ========================================
                # AdamW update
                # ========================================
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                exp_avg_hat = exp_avg / (1 - beta1 ** step_count)
                exp_avg_sq_hat = exp_avg_sq / (1 - beta2 ** step_count)

                update_adamw = exp_avg_hat / (exp_avg_sq_hat.sqrt() + eps)

                # ========================================
                # Homotopy blending + decoupled weight decay
                # ========================================
                blended = ((1 - s) * current_tju_lr * update_tju
                           + s * current_a_lr * update_adamw)

                # Clip blended update norm as safety net
                blended_norm = blended.norm()
                max_update = 1.0
                if blended_norm > max_update:
                    blended.mul_(max_update / (blended_norm + 1e-12))

                if wd != 0:
                    effective_lr = (1 - s) * current_tju_lr + s * current_a_lr
                    p.mul_(1 - effective_lr * wd)

                p.add_(blended, alpha=-1.0)
                prev_update.copy_(blended)

        return loss