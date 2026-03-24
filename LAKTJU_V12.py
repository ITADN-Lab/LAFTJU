import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import math


class LAKTJU_V12(Optimizer):
    r"""
    LAKTJU-V12: Layer-wise Adaptive Kronecker-factored Trajectory Unified optimizer.

    Built on V8 (best-performing variant). Key improvements:
      1) Gradient Centralization: zero-mean gradients for Conv/Linear weights
      2) Adaptive KF strength scheduling: cosine warm-up of KF contribution
      3) Improved TJU path: adaptive KF clip (grad-norm-ratio based)
      4) Unified decoupled weight decay (both TJU and AdamW paths)
      5) SAM-friendly design with KF hook management

    Architecture: TJU path (QGS + KF-PTC) blended with AdamW path via homotopy s(t).
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
            homotopy_speed=2.0,
            warmup=20,
            total_steps=0,
            kf_update_interval=20,
            kf_ema=0.95,
            kf_damping=1e-3,
            kf_warmup=500,
            kf_clip_max=50.0,
            grad_centralization=True,
    ):
        defaults = dict(
            tju_lr=tju_lr, a_lr=a_lr, beta1=beta1, beta2=beta2, eps=eps,
            weight_decay=weight_decay, c_base=c_base, kappa=kappa,
            homotopy_sharpness=homotopy_sharpness, homotopy_speed=homotopy_speed,
            warmup=warmup, total_steps=total_steps,
            kf_update_interval=kf_update_interval, kf_ema=kf_ema,
            kf_damping=kf_damping, kf_warmup=kf_warmup,
            kf_clip_max=kf_clip_max, grad_centralization=grad_centralization,
        )
        self.total_steps = total_steps
        self.c_base = c_base
        self.kappa = kappa
        self.homotopy_sharpness = homotopy_sharpness
        self.homotopy_speed = homotopy_speed
        self.kf_update_interval = kf_update_interval
        self.kf_ema = kf_ema
        self.kf_damping = kf_damping
        self.kf_warmup = kf_warmup
        self.kf_clip_max = kf_clip_max
        self.grad_centralization = grad_centralization

        # Global state
        self._global_step = 0

        # Kronecker factor storage
        self._kf_A = {}
        self._kf_G = {}
        self._kf_A_inv = {}
        self._kf_G_inv = {}
        self._kf_hooks = []
        self._param_to_module = {}

        # QGS loss amplification
        self._current_loss = None

        # SAM hook control
        self._skip_kf = False

        # Diagnostics
        self._diag = {}

        super(LAKTJU_V12, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LAKTJU_V12, self).__setstate__(state)

    def set_loss(self, loss_val):
        """Set current loss value for QGS loss amplification."""
        self._current_loss = loss_val

    def get_diagnostics(self):
        """Return diagnostics dict for logging."""
        return self._diag.copy()

    # ==================================================================
    # Gradient Centralization
    # ==================================================================
    @staticmethod
    def _centralize_gradient(grad):
        """Zero-mean gradient for Conv/Linear weights (≥2D tensors).
        Centralizes along all dims except the output dim (dim 0).
        """
        if grad.dim() > 1:
            # Mean over all dims except output channel (dim 0)
            dims = list(range(1, grad.dim()))
            grad.sub_(grad.mean(dim=dims, keepdim=True))
        return grad

    # ==================================================================
    # KF Strength Scheduling
    # ==================================================================
    def _kf_strength(self):
        """Cosine warm-up of KF contribution strength.
        Returns value in [0, 1]: 0 during early training, ramps to 1.
        """
        if self._global_step < self.kf_warmup:
            # Cosine warm-up: 0 -> 1 over kf_warmup steps
            return 0.5 * (1.0 - math.cos(math.pi * self._global_step / self.kf_warmup))
        return 1.0

    # ==================================================================
    # Kronecker Factor Hooks (same as V8, proven architecture)
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
        """Capture input activations -> compute A_l = E[aa^T]."""
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
        """Capture output gradients -> compute G_l = E[delta*delta^T]."""
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
        """Unfold conv input and spatial-average to get (B, C_in*k*k)."""
        k = module.kernel_size
        s = module.stride
        p = module.padding
        a_unf = torch.nn.functional.unfold(a, kernel_size=k, stride=s, padding=p)
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

    def _kf_precondition(self, p, grad_ema, c_l):
        """Apply KF preconditioning with adaptive clip."""
        mid = self._param_to_module[p.data_ptr()]
        G_inv = self._kf_G_inv[mid]
        A_inv = self._kf_A_inv[mid]

        d_out = G_inv.size(0)
        if isinstance(mid, nn.Conv2d):
            d_in_kf = A_inv.size(0)
            if mid.bias is not None:
                grad_mat = grad_ema.reshape(d_out, -1)
                if grad_mat.size(1) < d_in_kf:
                    grad_mat = torch.cat([grad_mat,
                        torch.zeros(d_out, 1, device=grad_mat.device, dtype=grad_mat.dtype)], dim=1)
            else:
                grad_mat = grad_ema.reshape(d_out, -1)
        elif isinstance(mid, nn.Linear):
            if mid.bias is not None and grad_ema.dim() == 1 and grad_ema.size(0) == d_out:
                return None
            grad_mat = grad_ema.reshape(d_out, -1)
            if grad_mat.size(1) < A_inv.size(0):
                grad_mat = torch.cat([grad_mat,
                    torch.zeros(d_out, A_inv.size(0) - grad_mat.size(1),
                                device=grad_mat.device, dtype=grad_mat.dtype)], dim=1)
        else:
            return None

        precond = c_l * (G_inv @ grad_mat @ A_inv)

        orig_numel = grad_ema.numel()
        precond_flat = precond.reshape(-1)[:orig_numel]
        result = precond_flat.reshape(grad_ema.shape)

        # Adaptive clip: based on grad norm ratio, capped at kf_clip_max
        grad_norm = grad_ema.norm()
        result_norm = result.norm()
        clip_ratio = self.kf_clip_max
        if result_norm > clip_ratio * grad_norm + 1e-8:
            result = result * (clip_ratio * grad_norm / (result_norm + 1e-8))

        return result

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

        # Periodically recompute KF inverses (after initial warmup)
        if (self._global_step % self.kf_update_interval == 0
                and self._global_step > 100
                and len(self._kf_A) > 0):
            self._update_kf_inverses()

        # KF strength for this step
        kf_s = self._kf_strength()

        # Diagnostics accumulators
        tju_update_norm = 0.0
        adam_update_norm = 0.0
        kf_active_count = 0
        total_param_count = 0
        gc_applied_count = 0

        for group in self.param_groups:
            tju_lr = group['tju_lr']
            a_lr = group['a_lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            wd = group['weight_decay']
            warmup_steps = group['warmup']

            # Homotopy schedule: tanh-based transition TJU -> AdamW
            progress = self._global_step / max(self.total_steps, 1)
            s = math.tanh(progress * self.homotopy_speed)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                total_param_count += 1

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

                # Warmup LR
                if step_count <= warmup_steps:
                    lr_scale = step_count / warmup_steps
                    current_tju_lr = tju_lr * lr_scale
                    current_a_lr = a_lr * lr_scale
                else:
                    current_tju_lr = tju_lr
                    current_a_lr = a_lr

                # ========================================
                # Gradient Centralization (Innovation #1)
                # ========================================
                if self.grad_centralization and grad.dim() > 1:
                    self._centralize_gradient(grad)
                    gc_applied_count += 1

                # ========================================
                # TJU path: QGS + KF-PTC (Core Innovation)
                # ========================================
                exp_avg_grad = state['exp_avg_grad']
                approx_hessian = state['approx_hessian']
                prev_update = state['prev_update']

                # TJU uses raw gradient (decoupled WD applied at blend step)
                grad_tju = grad.clone()

                bias_corr = 1 - (beta1 ** step_count)
                effective_alpha = (1 - beta1) / bias_corr

                grad_diff = grad_tju - exp_avg_grad
                exp_avg_grad.add_(grad_diff, alpha=effective_alpha)

                # KF preconditioning for Conv2d/Linear (Innovation #2: KF-PTC)
                update_tju = None
                if self._has_kf(p) and p.dim() >= 2 and kf_s > 0:
                    kf_update = self._kf_precondition(p, exp_avg_grad, 1.0)
                    if kf_update is not None:
                        # Blend KF with diagonal fallback based on kf_strength
                        if kf_s < 1.0:
                            # Also compute diagonal fallback
                            diag_update = self._diagonal_precondition(
                                exp_avg_grad, approx_hessian, prev_update,
                                grad_diff, effective_alpha)
                            update_tju = kf_s * kf_update + (1 - kf_s) * diag_update
                        else:
                            update_tju = kf_update
                        kf_active_count += 1

                # Diagonal fallback for 1D params or if KF not available
                if update_tju is None:
                    update_tju = self._diagonal_precondition(
                        exp_avg_grad, approx_hessian, prev_update,
                        grad_diff, effective_alpha)

                # ========================================
                # AdamW path
                # ========================================
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                exp_avg_hat = exp_avg / (1 - beta1 ** step_count)
                exp_avg_sq_hat = exp_avg_sq / (1 - beta2 ** step_count)

                update_adamw = exp_avg_hat / (exp_avg_sq_hat.sqrt() + eps)

                # ========================================
                # Homotopy blending + decoupled WD (both paths)
                # ========================================
                blended = ((1 - s) * current_tju_lr * update_tju
                           + s * current_a_lr * update_adamw)

                # Unified decoupled weight decay (Innovation #4)
                if wd != 0:
                    effective_lr = (1 - s) * current_tju_lr + s * current_a_lr
                    blended.add_(p, alpha=wd * effective_lr)

                p.add_(blended, alpha=-1.0)
                prev_update.copy_(blended)

                # Diagnostics
                tju_update_norm += update_tju.norm().item()
                adam_update_norm += update_adamw.norm().item()

        # Store diagnostics
        self._diag = {
            's': s if 's' in dir() else 0.0,
            'kf_strength': kf_s,
            'tju_update_norm': tju_update_norm,
            'adam_update_norm': adam_update_norm,
            'kf_active_count': kf_active_count,
            'total_param_count': total_param_count,
            'gc_applied_count': gc_applied_count,
        }

        return loss

    def _diagonal_precondition(self, exp_avg_grad, approx_hessian, prev_update,
                                grad_diff, effective_alpha):
        """Diagonal Hessian fallback (same as V8)."""
        rebound_thresh = 0.01
        diag_eps = 1e-4 / max(rebound_thresh, 1e-8)

        prev_norm = prev_update.norm(p=4).add_(diag_eps)
        prev_update_normalized = prev_update / prev_norm
        prev_update_sq = prev_update_normalized * prev_update_normalized

        delta_term = (
            (grad_diff / prev_norm * prev_update_normalized).sum()
            * (-effective_alpha)
            - (approx_hessian * prev_update_sq).sum()
        )
        approx_hessian.addcmul_(prev_update_sq, delta_term)

        denom_h = approx_hessian.abs().clamp_(min=rebound_thresh)
        return exp_avg_grad / denom_h
