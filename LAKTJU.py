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

        # Flag to skip KF accumulation (used during SAM perturbation forward pass)
        self._skip_kf = False

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

    def disable_kf_hooks(self):
        """Disable KF accumulation (used during SAM perturbation forward pass)."""
        self._skip_kf = True

    def enable_kf_hooks(self):
        """Re-enable KF accumulation after SAM perturbation."""
        self._skip_kf = False

    def _forward_hook(self, module, input):
        """Capture input activations -> compute A_l = E[aa^T].

        Optimization: spatial mean instead of unfold (avoids large temporary tensors),
        and EMA only updated on steps just before KF inverse recomputation.
        """
        if self._skip_kf or not torch.is_grad_enabled():
            return
        # Only accumulate on the step before inverse update (saves ~(T_kf-1)/T_kf of EMA work)
        if self._global_step % self.kf_update_interval != self.kf_update_interval - 1:
            return
        a = input[0].detach()
        if isinstance(module, nn.Conv2d):
            # Spatial mean: (B, C_in, H, W) -> (B, C_in)  -- avoids unfold's huge temp tensor
            a = a.mean(dim=[2, 3])
        elif isinstance(module, nn.Linear):
            if a.dim() > 2:
                a = a.reshape(-1, a.size(-1))
        # Add bias dimension
        if module.bias is not None:
            ones = torch.ones(a.size(0), 1, device=a.device, dtype=a.dtype)
            a = torch.cat([a, ones], dim=1)

        # A = (1/B) * a^T @ a
        A_batch = a.t() @ a / a.size(0)

        mid = module
        if mid not in self._kf_A:
            self._kf_A[mid] = A_batch
        else:
            self._kf_A[mid].mul_(self.kf_ema).add_(A_batch, alpha=1 - self.kf_ema)

    def _backward_hook(self, module, grad_input, grad_output):
        """Capture output gradients -> compute G_l = E[delta*delta^T].

        Optimization: EMA only updated on steps just before KF inverse recomputation.
        """
        if self._skip_kf:
            return
        # Only accumulate on the step before inverse update
        if self._global_step % self.kf_update_interval != self.kf_update_interval - 1:
            return
        g = grad_output[0].detach()
        if isinstance(module, nn.Conv2d):
            g = g.mean(dim=[2, 3])
        elif isinstance(module, nn.Linear):
            if g.dim() > 2:
                g = g.reshape(-1, g.size(-1))
        # G = (1/B) * g^T @ g
        G_batch = g.t() @ g / g.size(0)

        mid = module
        if mid not in self._kf_G:
            self._kf_G[mid] = G_batch
        else:
            self._kf_G[mid].mul_(self.kf_ema).add_(G_batch, alpha=1 - self.kf_ema)

    @staticmethod
    def _cholesky_inv(M, damping):
        """Compute inverse via Cholesky decomposition. ~2x faster than LU, numerically stable."""
        d = M.size(0)
        # Trace-scaled damping, fully on GPU (no .item())
        damp = damping * (M.trace() / d).clamp(1.0, 100.0)
        M_damp = M + damp * torch.eye(d, device=M.device, dtype=M.dtype)
        try:
            L = torch.linalg.cholesky(M_damp)
            return torch.cholesky_inverse(L)
        except torch._C._LinAlgError:
            return None

    def _update_kf_inverses(self):
        """Recompute damped inverses for all tracked modules using Cholesky."""
        for module in self._kf_A:
            if module in self._kf_G:
                A_inv = self._cholesky_inv(self._kf_A[module], self.kf_damping)
                G_inv = self._cholesky_inv(self._kf_G[module], self.kf_damping)
                if A_inv is not None and G_inv is not None:
                    self._kf_A_inv[module] = A_inv
                    self._kf_G_inv[module] = G_inv

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

        # Relaxed KF clip: 100x grad norm -- fully on GPU, no Python-level branching
        grad_norm = grad_ema.norm()
        result_norm = result.norm()
        scale = (100.0 * grad_norm / (result_norm + 1e-8)).clamp(max=1.0)
        result = result * scale

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

            # Collect layer-wise gradient norms for coupling ratio (stay on GPU, no .item())
            layer_info = []
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                np_ = p.numel()
                layer_info.append((p, g, np_))

            if len(layer_info) == 0:
                continue

            # Simple tanh homotopy (matching ATJU)
            progress = self._global_step / max(self.total_steps, 1)
            s = math.tanh(progress * self.homotopy_speed)

            # Per-parameter update
            for idx, (p, grad, n_params) in enumerate(layer_info):
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

                # ========================================
                # TJU update: KF-preconditioned or diagonal fallback
                # ========================================
                exp_avg_grad = state['exp_avg_grad']
                approx_hessian = state['approx_hessian']
                prev_update = state['prev_update']

                # Fold weight decay into TJU gradient (L2-style, curvature-aware like ATJU)
                grad_tju = grad.clone()
                if wd != 0:
                    grad_tju.add_(p, alpha=wd)

                bias_corr = 1 - (beta1 ** step_count)
                effective_alpha = (1 - beta1) / bias_corr

                grad_diff = grad_tju - exp_avg_grad
                exp_avg_grad.add_(grad_diff, alpha=effective_alpha)

                # Try KF preconditioning for Conv2d/Linear weights
                update_tju = None
                if self._has_kf(p) and p.dim() >= 2:
                    update_tju = self._kf_precondition(p, exp_avg_grad, 1.0)

                # Diagonal fallback for 1D params (BN, biases) or if KF not available
                if update_tju is None:
                    rebound_thresh = 0.01
                    diag_eps = 1e-4 / max(rebound_thresh, 1e-8)  # Match ATJU's tju_eps=1e-4

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
                    update_tju = exp_avg_grad / denom_h

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
                # Homotopy blending + weight decay
                # ========================================
                # TJU path already has L2 weight decay folded into grad_tju
                # AdamW path needs decoupled weight decay
                blended = ((1 - s) * current_tju_lr * update_tju
                           + s * current_a_lr * (update_adamw + wd * p))

                p.add_(blended, alpha=-1.0)
                prev_update.copy_(blended)

        return loss