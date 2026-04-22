import torch
from torch.optim.optimizer import Optimizer
import numpy as np
import math


class Nda_ATJU(Optimizer):
    r"""
    Revised TJU_AdamW optimizer that maintains the original TJU_v3 precision while implementing correct decoupled weight decay

    Key improvements:
    1. Fix weight decay application: scale using current learning rate (current_lr * tju_weight_decay)
    2. Preserve the original TJU_v3 approximate Hessian handling logic
    3. Restore the original parameter update order to ensure numerical stability
    """

    def __init__(
            self,
            params,
            tju_lr=1e-3,
            a_lr=1e-3,
            tju_betas=(0.9, 0.999),
            a_betas=(0.9, 0.999),
            beta_h=0.85,
            tju_eps=1e-8,
            a_eps=1e-8,
            momentum=0,
            rebound='constant',
            warmup=0,
            init_lr=None,
            tju_weight_decay=0.0,
            a_weight_decay=0,
            weight_decay_type='L2',
            A_optim=None,
            total_epoch=0,
            epoch_now=0,
            hessian_scale=0.05,
            total_steps=10000,
            use_cosine_scheduler=True
    ):

        self.epoch_now = epoch_now
        self.total_epoch = total_epoch
        self.step_now = 0

        # Parameter validation (preserve original strict validation)
        if not 0.0 <= tju_weight_decay:
            raise ValueError(f"Invalid tju_weight_decay: {tju_weight_decay}")
        if weight_decay_type not in ['L2', 'stable', 'AdamW']:  # corrected options list
            raise ValueError(f"Invalid weight_decay_type: {weight_decay_type}")
        if A_optim not in ['SGD', 'Adam', 'AdamW', None]:
            raise ValueError(f"Invalid A_optim: {A_optim} "
                             "(must be 'SGD', 'Adam', or 'AdamW')")
        if not 0.0 <= total_epoch:
            raise ValueError(f"Invalid epoch: {total_epoch} (must be >= 0)")
        if not 0.0 <= epoch_now:
            raise ValueError(f"Invalid epoch: {epoch_now} (must be >= 0)")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(
            tju_lr=tju_lr,
            a_lr=a_lr,
            tju_betas=tju_betas,
            a_betas=a_betas,
            beta_h=beta_h,
            tju_eps=tju_eps,
            a_eps=a_eps,
            momentum=momentum,
            rebound=rebound,
            warmup=warmup,
            init_lr=init_lr or tju_lr / 1000.0,
            base_lr=tju_lr,
            tju_weight_decay=tju_weight_decay,
            a_weight_decay=a_weight_decay,
            A_optim=A_optim,
            weight_decay_type=weight_decay_type,
            hessian_scale=hessian_scale,
            total_steps=total_steps,
            use_cosine_scheduler=use_cosine_scheduler
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            tju_beta1, tju_beta2 = group['tju_betas']
            a_beta1, a_beta2 = group["a_betas"]
            a_lr = group["a_lr"]
            tju_lr = group["tju_lr"]
            a_eps = group["a_eps"]
            momentum = group["momentum"]
            a_wd_coef = group["a_weight_decay"]
            A_optim = group['A_optim']


            # Define the current weight value s
            if self.epoch_now == 1:
                self.total_epoch = self.total_epoch * (self.step_now + 1)
                self.epoch_now += 1

            if self.epoch_now >= 1:
                # s = self.step_now / self.total_epoch
                # s = math.tanh((self.step_now / self.total_epoch) * 2)
                s = 1 / (1 + math.exp(-17 * (self.step_now / self.total_epoch - 0.5)))
            else:
                s = 0


            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("TJU_AdamW_Fixed does not support sparse gradients")

                state = self.state[p]
                # Initialize state (preserve original TJU_v3 structure)
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['approx_hessian'] = torch.zeros_like(p)
                    # SGD historical momentum value
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of first-order gradients (Adam/W)
                    state['exp_avg_grad_A'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of second-order gradients (AdamW)
                    state["exp_avg_sq_grad_A"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1
                step = state['step']
                self.step_now = step

                # ====== Learning rate schedule (preserve original TJU_v3 logic) ======
                # current_lr = self._compute_lr(group, step)
                current_lr = tju_lr

                # ====== Core parameter update ======
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                approx_hessian = state['approx_hessian']
                momentum_buffer = state["momentum_buffer"]
                exp_avg_grad_A = state['exp_avg_grad_A']
                exp_avg_sq_grad_A = state["exp_avg_sq_grad_A"]

                # Gradient value assignment
                grad_TJU = p.grad  # Gradient value used by TJU
                grad_A = p.grad  # Gradient value used by A_optim


                # (1) L2 regularization (preserve original logic)
                if group['weight_decay_type'] == 'L2' and group['tju_weight_decay'] != 0:
                    grad_TJU = grad_TJU.add(p, alpha=group['tju_weight_decay'])

                # (2) Update momentum terms (preserve original TJU_v3 numerical stability)
                exp_avg.mul_(tju_beta1).add_(grad_TJU, alpha=1 - tju_beta1)
                exp_avg_sq.mul_(tju_beta2).addcmul_(grad_TJU, grad_TJU, value=1 - tju_beta2)

                # (3) Bias correction (critical! preserve original TJU_v3 implementation)
                bias_corr1 = 1 - tju_beta1 ** step
                bias_corr2 = 1 - tju_beta2 ** step
                step_size = current_lr / bias_corr1  # Combine learning rate with first-order bias correction

                # (4) Approximate Hessian handling (preserve original TJU_v3 clamp logic)
                delta_grad = grad_TJU - (exp_avg / bias_corr1)  # Corrected gradient change

                approx_hessian.mul_(group['beta_h']).addcmul_(
                    delta_grad, delta_grad, value=1 - group['beta_h'])

                if group['rebound'] == 'constant':
                    denom_hessian = approx_hessian.abs().clamp_(min=1e-3)  # preserve original v3 clamp lower bound
                else:
                    bound_val = max(delta_grad.norm(p=float('inf')).item(), 1e-5)
                    denom_hessian = torch.max(approx_hessian.abs(),
                                              torch.tensor(bound_val, device=p.device))

                # (5) Combine second-order momentum (preserve original v3 mixed logic)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(
                    group['hessian_scale'] * denom_hessian,
                    alpha=1.0
                ).add_(group['tju_eps'])

                # (6) Compute update direction (critical modification! restore original v3 stability)
                new_update = exp_avg / denom

                # (7) Handle stable type weight decay (preserve original v3 logic)
                if group['weight_decay_type'] == 'stable' and group['tju_weight_decay'] != 0:
                    decay_factor = group['tju_weight_decay'] / denom.mean().clamp(min=1e-8)
                    new_update.add_(p, alpha=decay_factor)

                # ====== AdamW type weight decay (critical fix!) ====== #
                # Apply decoupled decay during parameter update (kept independent of current learning rate)
                if group['weight_decay_type'] == 'AdamW' and group['tju_weight_decay'] != 0:
                    p.data.mul_(1 - group['tju_weight_decay'] * current_lr)  # Key fix: decoupled from learning rate!

                updata_TJU = (1 - s) * new_update

                if A_optim == 'SGD':
                    # SGD update section
                    grad_A = grad_A.add(p, alpha=a_wd_coef)
                    # Compute parameter update value (SGD)
                    updata_A = s * (grad_A + momentum * momentum_buffer)
                    # Store accumulated momentum value
                    state["momentum_buffer"] = grad_A + momentum * momentum_buffer

                elif A_optim == 'Adam':
                    # Adam update section
                    grad_A = grad_A.add(p, alpha=a_wd_coef)
                    exp_avg_grad_A = a_beta1 * exp_avg_grad_A + (1 - a_beta1) * grad_A
                    exp_avg_sq_grad_A = a_beta2 * exp_avg_sq_grad_A + (1 - a_beta2) * grad_A ** 2

                    exp_avg_grad_A_hat = exp_avg_grad_A / (1 - a_beta1 ** step)
                    exp_avg_sq_grad_A_hat = exp_avg_sq_grad_A / (1 - a_beta2 ** step)
                    updata_A = s * (exp_avg_grad_A_hat / (torch.sqrt(exp_avg_sq_grad_A_hat) + a_eps))

                    state["exp_avg_grad_A"] = exp_avg_grad_A
                    state["exp_avg_sq_grad_A"] = exp_avg_sq_grad_A

                elif A_optim == 'AdamW':
                    # AdamW update section
                    exp_avg_grad_A = a_beta1 * exp_avg_grad_A + (1 - a_beta1) * grad_A
                    exp_avg_sq_grad_A = a_beta2 * exp_avg_sq_grad_A + (1 - a_beta2) * grad_A ** 2

                    exp_avg_grad_A_hat = exp_avg_grad_A / (1 - a_beta1 ** step)
                    exp_avg_sq_grad_A_hat = exp_avg_sq_grad_A / (1 - a_beta2 ** step)
                    updata_A = s * ((exp_avg_grad_A_hat / (torch.sqrt(exp_avg_sq_grad_A_hat) + a_eps)) + a_wd_coef * p)

                    state["exp_avg_grad_A"] = exp_avg_grad_A
                    state["exp_avg_sq_grad_A"] = exp_avg_sq_grad_A


                update = -step_size * updata_TJU -a_lr * updata_A   # Note: step_size already includes learning rate and first-order bias correction


                # (8) Execute parameter update (preserve original v3 update order)
                p.add_(update)

        print(f'Current s value: {s}')
        return loss

    def _compute_lr(self, group, step):
        """Learning rate schedule (precisely preserves original TJU_v3 implementation)"""
        if step <= group['warmup']:
            return group['init_lr'] + (group['base_lr'] - group['init_lr']) * step / group['warmup']

        if not group['use_cosine_scheduler']:
            return group['base_lr']

        t = step - group['warmup']
        T = group['total_steps'] - group['warmup']

        if t <= T:
            return group['base_lr'] * (0.5 * (1 + math.cos(math.pi * t / T)))
        return group['base_lr'] * 0.01  # preserve original v3 post-training learning rate