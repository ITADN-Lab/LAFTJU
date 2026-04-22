import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import math


class LAKTJU_Lite(Optimizer):
    r"""
    LAKTJU-Lite: O(P) time and memory variant of LAKTJU.

    Key idea: replace full Kronecker factor matrices A ∈ R^{n×n}, G ∈ R^{m×m}
    with their diagonal approximations stored as vectors a ∈ R^n, g ∈ R^m.

    For weight W ∈ R^{m×n}, the preconditioned gradient is:
        P^{-1} ∇W ≈ ∇W_{ij} / ((g_i + δ)(a_j + δ))

    This is equivalent to a separable (outer-product) adaptive step size,
    capturing row-wise (output channel) and column-wise (input channel)
    curvature independently. Memory: O(m+n) per layer vs O(m²+n²) for full KF.

    Additional savings vs LAKTJU:
    - No matrix inversion (no Cholesky, no O(C³) cost)
    - No A_inv / G_inv buffers
    - Hooks are O(C) per layer per step (just vector EMA)
    - Dual-path merged: TJU path reuses AdamW m1/m2 buffers (no extra state)

    Memory breakdown vs AdamW (params P, layers L with avg width C):
        AdamW:       2P  (m1, m2)
        LAKTJU:      5P + 4·L·C²  (m1,m2,m_tju,H,prev) + (A,G,A_inv,G_inv)
        LAKTJU-Lite: 2P + 2·L·C   (m1,m2) + (diag_a, diag_g)  ≈ 2P
    """

    def __init__(
            self,
            params,
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=2e-3,
            homotopy_speed=8.0,
            total_steps=0,
            warmup=100,
            kf_ema=0.95,
            kf_damping=1e-3,
            kf_update_interval=20,   # diagonal update is cheap but still skip for consistency
    ):
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, eps=eps,
            weight_decay=weight_decay, homotopy_speed=homotopy_speed,
            total_steps=total_steps, warmup=warmup,
            kf_ema=kf_ema, kf_damping=kf_damping,
            kf_update_interval=kf_update_interval,
        )
        self.total_steps = total_steps
        self.homotopy_speed = homotopy_speed
        self.kf_ema = kf_ema
        self.kf_damping = kf_damping
        self.kf_update_interval = kf_update_interval

        self._global_step = 0
        self._skip_kf = False

        # Diagonal KF vectors: keyed by module
        # diag_a[mod] ∈ R^{C_in}  (input channel variance)
        # diag_g[mod] ∈ R^{C_out} (output channel variance)
        self._diag_a = {}
        self._diag_g = {}
        self._kf_hooks = []
        self._param_to_module = {}

        super(LAKTJU_Lite, self).__init__(params, defaults)

    # ------------------------------------------------------------------ #
    # Hook registration                                                    #
    # ------------------------------------------------------------------ #
    def register_hooks(self, model):
        """Register lightweight diagonal-KF hooks on Conv2d and Linear layers."""
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                for p in module.parameters():
                    self._param_to_module[p.data_ptr()] = module
                h_fwd = module.register_forward_pre_hook(self._fwd_hook)
                h_bwd = module.register_full_backward_hook(self._bwd_hook)
                self._kf_hooks.extend([h_fwd, h_bwd])

    def disable_kf_hooks(self):
        self._skip_kf = True

    def enable_kf_hooks(self):
        self._skip_kf = False

    def _fwd_hook(self, module, input):
        """Accumulate diagonal of A = E[aa^T] as E[a_j^2] per input channel j."""
        if self._skip_kf or not torch.is_grad_enabled():
            return
        # Only update every kf_update_interval steps (still O(C) so even every step is fine,
        # but skipping reduces hook overhead when interval > 1)
        if self._global_step % self.kf_update_interval != 0:
            return
        a = input[0].detach()
        if isinstance(module, nn.Conv2d):
            # (B, C_in, H, W) -> mean over spatial -> (B, C_in)
            a = a.mean(dim=[2, 3])
        elif a.dim() > 2:
            a = a.reshape(-1, a.size(-1))
        # diag of A: E[a_j^2] = mean over batch of a_j^2
        # shape: (C_in,)
        diag_a_batch = (a * a).mean(dim=0)

        mid = module
        if mid not in self._diag_a:
            self._diag_a[mid] = diag_a_batch
        else:
            self._diag_a[mid].mul_(self.kf_ema).add_(diag_a_batch, alpha=1 - self.kf_ema)

    def _bwd_hook(self, module, grad_input, grad_output):
        """Accumulate diagonal of G = E[gg^T] as E[g_i^2] per output channel i."""
        if self._skip_kf:
            return
        if self._global_step % self.kf_update_interval != 0:
            return
        g = grad_output[0].detach()
        if isinstance(module, nn.Conv2d):
            # (B, C_out, H, W) -> mean over spatial -> (B, C_out)
            g = g.mean(dim=[2, 3])
        elif g.dim() > 2:
            g = g.reshape(-1, g.size(-1))
        # diag of G: E[g_i^2]
        # shape: (C_out,)
        diag_g_batch = (g * g).mean(dim=0)

        mid = module
        if mid not in self._diag_g:
            self._diag_g[mid] = diag_g_batch
        else:
            self._diag_g[mid].mul_(self.kf_ema).add_(diag_g_batch, alpha=1 - self.kf_ema)

    # ------------------------------------------------------------------ #
    # Preconditioning                                                      #
    # ------------------------------------------------------------------ #
    def _has_diag_kf(self, p):
        mid = self._param_to_module.get(p.data_ptr())
        return mid is not None and mid in self._diag_a and mid in self._diag_g

    def _diag_precondition(self, p, grad_ema):
        """
        Apply diagonal KF preconditioning.

        For weight W (C_out, C_in[*k*k]):
            precond_{ij} = grad_ema_{ij} / sqrt((g_i + δ)(a_j + δ))

        Using sqrt matches Adam's variance-normalization intuition and
        prevents over-aggressive scaling in early training.
        """
        mid = self._param_to_module[p.data_ptr()]
        damp = self.kf_damping

        diag_g = self._diag_g[mid]   # (C_out,)
        diag_a = self._diag_a[mid]   # (C_in,)

        if isinstance(mid, nn.Conv2d):
            # weight: (C_out, C_in, kH, kW) -> treat as (C_out, C_in)
            # broadcast: g (C_out,1,1,1) * a (1,C_in,1,1)
            g_vec = (diag_g + damp).rsqrt().view(-1, 1, 1, 1)
            a_vec = (diag_a + damp).rsqrt().view(1, -1, 1, 1)
            # handle bias-augmented diag_a (len = C_in+1)
            if a_vec.size(1) == grad_ema.size(1) + 1:
                a_vec = a_vec[:, :grad_ema.size(1)]
            elif a_vec.size(1) != grad_ema.size(1):
                return None
            precond = grad_ema * g_vec * a_vec

        elif isinstance(mid, nn.Linear):
            if grad_ema.dim() == 1:
                # bias parameter - skip KF
                return None
            # weight: (C_out, C_in)
            g_vec = (diag_g + damp).rsqrt().unsqueeze(1)   # (C_out, 1)
            a_vec = (diag_a + damp).rsqrt().unsqueeze(0)   # (1, C_in)
            if a_vec.size(1) == grad_ema.size(1) + 1:
                a_vec = a_vec[:, :grad_ema.size(1)]
            elif a_vec.size(1) != grad_ema.size(1):
                return None
            precond = grad_ema * g_vec * a_vec
        else:
            return None

        # Clip to 10x grad norm for stability (cheap: no sync needed)
        grad_norm = grad_ema.norm()
        result_norm = precond.norm()
        scale = (10.0 * grad_norm / (result_norm + 1e-8)).clamp(max=1.0)
        return precond * scale

    # ------------------------------------------------------------------ #
    # Main step                                                            #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1

        for group in self.param_groups:
            lr       = group['lr']
            beta1    = group['beta1']
            beta2    = group['beta2']
            eps      = group['eps']
            wd       = group['weight_decay']
            warmup   = group['warmup']
            hs       = group['homotopy_speed']
            T        = group['total_steps'] or self.total_steps

            # Homotopy: s=0 -> TJU-dominated, s=1 -> AdamW-dominated
            progress = self._global_step / max(T, 1)
            s = math.tanh(progress * hs)

            # LR warmup scale
            lr_scale = min(1.0, self._global_step / max(warmup, 1))
            cur_lr = lr * lr_scale

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg']    = torch.zeros_like(p)  # m1 (shared)
                    state['exp_avg_sq'] = torch.zeros_like(p)  # m2 (shared)

                state['step'] += 1
                t = state['step']
                m1 = state['exp_avg']
                m2 = state['exp_avg_sq']

                # ---- AdamW moments (standard) ----
                m1.mul_(beta1).add_(grad, alpha=1 - beta1)
                m2.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t
                m1_hat = m1 / bc1
                m2_hat = m2 / bc2

                # ---- AdamW update ----
                update_adamw = m1_hat / (m2_hat.sqrt() + eps)

                # ---- TJU path: diagonal-KF preconditioned gradient ----
                if self._has_diag_kf(p) and p.dim() >= 2:
                    update_tju = self._diag_precondition(p, m1_hat)
                else:
                    update_tju = None

                if update_tju is None:
                    # Fallback: normalized by Adam's v estimate (= AdamW-style)
                    update_tju = update_adamw

                # ---- Homotopy blend ----
                # Early: TJU path (curvature-aware), Late: AdamW path
                update = (1 - s) * update_tju + s * update_adamw

                # ---- Parameter update with decoupled weight decay ----
                p.add_(update, alpha=-cur_lr)
                if wd != 0:
                    p.mul_(1 - cur_lr * wd)

        return loss
