"""
LAKTJU V10: Layer-wise Adaptive Kronecker-factored Trajectory Unified optimizer.

V10 核心改进（基于V9实验分析）：
  V9的根本问题：SGD-momentum的update_main norm(~0.15)远小于AdamW的update_adamw norm(~43.9)，
  导致homotopy混合中主路径贡献不到1%，V9实质退化为受限AdamW。

  V10修复：
  1) Norm-matching：混合前将主路径更新缩放到与AdamW相同量级
     - 主路径提供方向（SGD-momentum + KF修正）
     - AdamW提供自适应步长参考
     - Homotopy参数s真正控制方向混合比例
  2) 统一学习率：只用一个lr，消除双lr的混乱
  3) 改进诊断：记录缩放后的实际贡献比
"""
import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import math


class LAKTJU_V10(Optimizer):
    r"""
    LAKTJU V10: Norm-matched SGD-momentum + AdamW homotopy blending.

    Key changes from V9:
      - Norm-matching: scale main path to match AdamW step size before blending
      - Single unified learning rate (no more tju_lr / a_lr split)
      - Homotopy s now truly controls direction blend ratio
    """

    def __init__(
            self,
            params,
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=1e-3,
            momentum=0.9,
            alpha_kf=0.05,
            alpha_adam_kf=0.0,
            s_max=0.7,
            homotopy_speed=5.0,
            warmup=100,
            total_steps=0,
            kf_update_interval=20,
            kf_ema=0.95,
            kf_damping=1e-3,
            grad_clip=0.0,
            diag_interval=50,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, eps=eps,
            weight_decay=weight_decay, momentum=momentum,
            alpha_kf=alpha_kf, alpha_adam_kf=alpha_adam_kf,
            s_max=s_max, homotopy_speed=homotopy_speed,
            warmup=warmup, total_steps=total_steps,
            kf_update_interval=kf_update_interval, kf_ema=kf_ema,
            kf_damping=kf_damping, grad_clip=grad_clip,
        )
        self.total_steps = total_steps
        self.homotopy_speed = homotopy_speed
        self.s_max = s_max
        self.kf_update_interval = kf_update_interval
        self.kf_ema = kf_ema
        self.kf_damping = kf_damping
        self.diag_interval = diag_interval

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

        super(LAKTJU_V10, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LAKTJU_V10, self).__setstate__(state)

    # ==================================================================
    # Kronecker Factor Hooks (same as V9)
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
                A_d = A + damping * torch.eye(A.size(0), device=A.device, dtype=A.dtype)
                G_d = G + damping * torch.eye(G.size(0), device=G.device, dtype=G.dtype)
                try:
                    self._kf_A_inv[module] = torch.linalg.inv(A_d)
                    self._kf_G_inv[module] = torch.linalg.inv(G_d)
                except Exception:
                    pass

    def _has_kf(self, p):
        ptr = p.data_ptr()
        if ptr not in self._param_to_module:
            return False
        module = self._param_to_module[ptr]
        return module in self._kf_A_inv and module in self._kf_G_inv

    def _kf_precondition(self, p, v):
        ptr = p.data_ptr()
        module = self._param_to_module[ptr]
        A_inv = self._kf_A_inv[module]
        G_inv = self._kf_G_inv[module]

        if isinstance(module, nn.Conv2d):
            v_2d = v.reshape(v.size(0), -1)
            precond = G_inv @ v_2d @ A_inv
            return precond.reshape_as(v)
        elif isinstance(module, nn.Linear):
            if module.bias is not None and v.shape == module.weight.shape:
                return G_inv @ v @ A_inv[:v.size(1), :v.size(1)]
            return G_inv @ v @ A_inv
        return None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1

        # Update KF inverses periodically
        if (self._global_step % self.kf_update_interval == 0
                and len(self._kf_A) > 0):
            self._update_kf_inverses()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            wd = group['weight_decay']
            mom = group['momentum']
            alpha_kf = group['alpha_kf']
            alpha_adam_kf = group['alpha_adam_kf']
            warmup_steps = group['warmup']
            grad_clip_val = group['grad_clip']

            # Homotopy: s = s_max * tanh(progress * speed)
            progress = self._global_step / max(self.total_steps, 1)
            s_raw = math.tanh(progress * self.homotopy_speed)
            s = self.s_max * s_raw

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                step_count = state['step']

                # Warmup learning rate
                if step_count <= warmup_steps:
                    lr_scale = step_count / warmup_steps
                    current_lr = lr * lr_scale
                else:
                    current_lr = lr

                # Optional gradient clipping (per-param)
                if grad_clip_val > 0:
                    gn = grad.norm()
                    if gn > grad_clip_val:
                        grad = grad * (grad_clip_val / (gn + 1e-12))

                # ========================================
                # Path 1: SGD-momentum + KF direction correction
                # ========================================
                buf = state['momentum_buffer']
                buf.mul_(mom).add_(grad)

                # KF direction correction on momentum buffer
                if self._has_kf(p) and p.dim() >= 2:
                    kf_dir = self._kf_precondition(p, buf)
                    if kf_dir is not None:
                        update_main = (1 - alpha_kf) * buf + alpha_kf * kf_dir
                    else:
                        update_main = buf
                else:
                    update_main = buf

                # ========================================
                # Path 2: AdamW
                # ========================================
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                exp_avg_hat = exp_avg / (1 - beta1 ** step_count)
                exp_avg_sq_hat = exp_avg_sq / (1 - beta2 ** step_count)

                update_adamw = exp_avg_hat / (exp_avg_sq_hat.sqrt() + eps)

                # KF correction on AdamW direction
                if alpha_adam_kf > 0 and self._has_kf(p) and p.dim() >= 2:
                    kf_adam_dir = self._kf_precondition(p, update_adamw)
                    if kf_adam_dir is not None:
                        update_adamw = (1 - alpha_adam_kf) * update_adamw + alpha_adam_kf * kf_adam_dir

                # ========================================
                # V10 核心：Norm-matched homotopy blending
                # ========================================
                main_norm = update_main.norm()
                adam_norm = update_adamw.norm()

                # Scale main path to match AdamW's adaptive step size
                if main_norm > eps:
                    update_main_scaled = update_main * (adam_norm / main_norm)
                else:
                    update_main_scaled = update_main

                # Now blend directions with matched magnitudes
                # s controls direction: (1-s)*SGD_dir + s*AdamW_dir
                blended = (1 - s) * update_main_scaled + s * update_adamw

                # Decoupled weight decay
                if wd != 0:
                    p.mul_(1.0 - current_lr * wd)

                # Apply update with unified learning rate
                p.add_(blended, alpha=-current_lr)

                # ========================================
                # Diagnostics
                # ========================================
                if self.diag_interval > 0 and self._global_step % self.diag_interval == 0:
                    if not self._diagnostics:
                        self._diagnostics = {
                            's': s,
                            'progress': progress,
                            'main_update_norm': 0.0,
                            'adam_update_norm': 0.0,
                            'main_scaled_norm': 0.0,
                            'blended_norm': 0.0,
                            'kf_active_count': 0,
                            'total_param_count': 0,
                        }
                    self._diagnostics['main_update_norm'] += main_norm.item()
                    self._diagnostics['adam_update_norm'] += adam_norm.item()
                    self._diagnostics['main_scaled_norm'] += update_main_scaled.norm().item()
                    self._diagnostics['blended_norm'] += blended.norm().item()
                    self._diagnostics['total_param_count'] += 1
                    if self._has_kf(p) and p.dim() >= 2:
                        self._diagnostics['kf_active_count'] += 1

        return loss

    def get_diagnostics(self):
        """Return and clear diagnostics dict."""
        d = self._diagnostics.copy()
        self._diagnostics = {}
        return d
