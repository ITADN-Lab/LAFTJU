import numpy as np
import torch
from torch.optim.optimizer import Optimizer
import math


class ATJU(Optimizer):
    r"""
    TJU_v1: A variant of Atom/TJU-based optimizer for deep learning.

    This optimizer integrates approximate Hessian information with an EMA of gradients
    to guide parameter updates. It supports various weight decay strategies and warmup
    schedules for more flexible training.

    Args:
        params (iterable):
            Model parameters to optimize (can be a single iterable or multiple param groups).
        lr (float):
            Base learning rate for updates.
        beta (float, optional):
            Momentum factor for gradient EMA. Default: 0.9
        eps (float, optional):
            A small constant for numerical stability (denominator addition). Default: 1e-4
        rebound (str, optional):
            Mode for bounding the diagonal Hessian. {'constant', 'belief'}. Default: 'constant'
        warmup (int, optional):
            Number of warmup steps during which the learning rate ramps from init_lr to lr. Default: 500
        init_lr (float, optional):
            Learning rate used at start of warmup. Default: lr/1000
        weight_decay (float, optional):
            Weight decay coefficient. Default: 0
        weight_decay_type (str, optional):
            Weight decay type: {'L2', 'decoupled', 'stable'}.
            If unset, defaults to 'L2' for 'constant' rebound or 'decoupled' otherwise.
    """

    def __init__(
            self,
            params,
            lr,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            momentum=0,
            rebound='constant',
            warmup=20,
            init_lr=None,
            weight_decay=0,
            weight_decay_type=None,
            A_optim=None,
            total_epoch=0,
            epoch_now=0
    ):

        self.epoch_now = epoch_now
        self.total_epoch = total_epoch
        self.step_now = 0
        # -- Parameter validity check --
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate value: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1} (must be in [0.0, 1.0))")
        if rebound not in ['constant', 'belief']:
            raise ValueError(f"Invalid rebound mode: {rebound}, must be 'constant' or 'belief'")
        if not 0 <= warmup:
            raise ValueError(f"Invalid warmup steps: {warmup} (must be >= 0)")
        if not 0.0 <= total_epoch:
            raise ValueError(f"Invalid epoch: {total_epoch} (must be >= 0)")
        if not 0.0 <= epoch_now:
            raise ValueError(f"Invalid epoch: {epoch_now} (must be >= 0)")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")

        # If init_lr is not provided, default to init_lr = lr / 1000
        if init_lr is None:
            init_lr = lr / 1000
        if not 0.0 <= init_lr <= lr:
            raise ValueError(f"Invalid init_lr: {init_lr} (must be in [0, lr])")

        # Weight decay coefficient and mode
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay} (must be >= 0)")
        if weight_decay_type is None:
            # If rebound='constant', default to L2; otherwise use decoupled
            weight_decay_type = 'L2' if rebound == 'constant' else 'decoupled'
        if weight_decay_type not in ['L2', 'decoupled', 'stable']:
            raise ValueError(f"Invalid weight_decay_type: {weight_decay_type} "
                             "(must be 'L2', 'decoupled', or 'stable')")
        if A_optim not in ['SGD', 'Adam', 'AdamW', None]:
            raise ValueError(f"Invalid A_optim: {A_optim} "
                             "(must be 'SGD', 'Adam', or 'AdamW')")

        # Store default hyperparameters in a unified dictionary
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            momentum=momentum,
            rebound=rebound,
            warmup=warmup,
            init_lr=init_lr,
            base_lr=lr,
            weight_decay=weight_decay,
            weight_decay_type=weight_decay_type,
            A_optim=A_optim,
        )
        # ATJU inherits all attributes from the parent class, corresponding to parent params, defaults
        super(ATJU, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ATJU, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional):
                A closure that reevaluates the model and returns the loss.

        Returns:
            loss (float, optional): The loss from closure if provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Iterate over each param group in order
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            momentum = group["momentum"]
            rebound_mode = group['rebound']
            warmup_steps = group['warmup']
            init_lr = group['init_lr']
            base_lr = group['base_lr']
            wd_coef = group['weight_decay']
            wd_type = group['weight_decay_type']
            A_optim = group['A_optim']

            # Define the current weight value s
            if self.epoch_now == 1:
                self.total_epoch = self.total_epoch * (self.step_now + 1)
                self.epoch_now += 1

            if self.epoch_now >= 1:
                s = self.step_now / self.total_epoch
            else:
                s = 0

            # Iterate over parameters in this group
            for p in group['params']:
                if p.grad is None:
                    continue

                # Get the state dictionary for the current parameter
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    # SGD historical momentum value
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of first-order gradients (TJU)
                    state['exp_avg_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of first-order gradients (Adam/W)
                    state['exp_avg_grad_A'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of second-order gradients (AdamW)
                    state["exp_avg_sq_grad_A"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of the approximate Hessian
                    state['approx_hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Previous update direction
                    state['prev_update'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                step_count = state['step']
                if step_count < warmup_steps:
                    # Linearly ramp from init_lr to base_lr
                    current_lr = (base_lr - init_lr) * (step_count / warmup_steps) + init_lr
                else:
                    current_lr = lr

                step_count += 1
                state['step'] = step_count
                self.step_now = step_count

                grad = p.grad  # raw gradient value
                grad_TJU = p.grad  # gradient value used by TJU
                grad_A = p.grad  # gradient value used by A_optim

                if grad.is_sparse:
                    raise RuntimeError("ATJU does not support sparse gradients.")

                #   TJU update section
                # If L2 decay, add the decay term directly to the gradient
                if wd_coef != 0 and wd_type == 'L2':
                    grad_TJU = grad_TJU.add(p, alpha=wd_coef)

                exp_avg_grad = state['exp_avg_grad']
                momentum_buffer = state["momentum_buffer"]
                exp_avg_grad_A = state['exp_avg_grad_A']
                exp_avg_sq_grad_A = state["exp_avg_sq_grad_A"]
                approx_hessian = state['approx_hessian']
                prev_update = state['prev_update']

                bias_corr = 1 - (beta1 ** step_count)
                effective_alpha = (1 - beta1) / bias_corr

                # Compute gradient difference
                grad_diff = grad_TJU - exp_avg_grad  # current parameter gradient minus the stored historical gradient

                # Select threshold based on rebound mode
                if rebound_mode == 'belief':
                    rebound_thresh = grad_diff.norm(p=np.inf)  # dynamically set threshold based on the max absolute value of the current gradient difference (grad_diff)
                else:
                    rebound_thresh = 0.01
                    eps = eps / max(rebound_thresh, 1e-8)  # prevent division by zero

                # (1) Update first-order gradient mean: exp_avg_grad <- exp_avg_grad + effective_alpha * grad_diff
                exp_avg_grad.add_(grad_diff, alpha=effective_alpha)

                # (2) Normalize the previous update, used to compute the approximate Hessian update
                prev_update_norm = prev_update.norm(p=4).add(eps)  # compute L4 norm and add eps to prevent zero denominator
                prev_update.div_(prev_update_norm)  # normalize prev_update in-place by its L4 norm
                prev_update_sq = prev_update.mul(prev_update)  # compute element-wise square of the normalized prev_update

                # Compute delta, used to update the Hessian
                # inner product of grad_diff / prev_update_norm with prev_update
                delta_term = (grad_diff.div_(prev_update_norm)  # normalize grad_diff
                              .mul_(prev_update)  # compute element-wise product of normalized grad_diff and historical update prev_update
                              .sum()  # sum over all parameter dimensions to get a scalar
                              .mul_(-effective_alpha)  # multiply result by negative effective learning rate effective_alpha
                              ) - approx_hessian.mul(
                    prev_update_sq).sum()  # multiply approximate Hessian approx_hessian by squared historical update prev_update_sq and sum
                # subtract the two terms to get delta_term

                # (3) Update Hessian: approx_hessian <- approx_hessian + (delta_term * prev_update_sq)
                approx_hessian.addcmul_(prev_update_sq, delta_term)

                # (4) Compute new update direction new_update
                if rebound_mode == 'belief':
                    denom_h = torch.max(approx_hessian.abs(),
                                        rebound_thresh)  # element-wise maximum of absolute value of approx_hessian and rebound_thresh
                    denom_h.add_(eps / effective_alpha)  # prevent division by zero
                else:
                    denom_h = approx_hessian.abs().clamp_(
                        min=rebound_thresh)  # take absolute value of approx_hessian and clamp values below rebound_thresh to rebound_thresh

                new_update = exp_avg_grad.div(denom_h)  # scale historical gradient momentum exp_avg_grad by inverse curvature to get the final update direction

                # (5) Apply decoupled or stable weight decay if needed
                if wd_coef != 0 and wd_type != 'L2':  # if weight decay is non-zero and mode is not L2 regularization
                    if wd_type == 'stable':
                        scaled_decay = wd_coef / max(denom_h.mean().item(), 1e-8)  # compute mean of denom_h (reflecting parameter curvature).
                        # divide wd_coef by this mean to get the scaled decay factor scaled_decay.
                        new_update.add_(p, alpha=scaled_decay)  # apply the scaled decay factor to parameter p, updating direction new_update
                    else:
                        # decoupled
                        new_update.add_(p, alpha=wd_coef)  # apply fixed-weight decay directly (independent of gradient)





                # (6) Compute parameter update value (TJU)
                updata_TJU = (1 - s) * new_update

                if A_optim == 'SGD':
                    # SGD update section
                    # Compute parameter update value (SGD)
                    updata_A = s * (grad_A + momentum * momentum_buffer)
                    # Store accumulated momentum value
                    state["momentum_buffer"] = grad_A + momentum * momentum_buffer

                elif A_optim == 'Adam':
                    # Adam update section
                    grad_A = grad_A.add(p, alpha=wd_coef)
                    exp_avg_grad_A = beta1 * exp_avg_grad_A + (1 - beta1) * grad_A
                    exp_avg_sq_grad_A = beta2 * exp_avg_sq_grad_A + (1 - beta2) * grad_A ** 2

                    exp_avg_grad_A_hat = exp_avg_grad_A / (1 - beta1 ** step_count)
                    exp_avg_sq_grad_A_hat = exp_avg_sq_grad_A / (1 - beta2 ** step_count)
                    updata_A = s * (exp_avg_grad_A_hat / (torch.sqrt(exp_avg_sq_grad_A_hat) + eps))

                    state["exp_avg_grad_A"] = exp_avg_grad_A
                    state["exp_avg_sq_grad_A"] = exp_avg_sq_grad_A

                elif A_optim == 'AdamW':
                    # AdamW update section
                    exp_avg_grad_A = beta1 * exp_avg_grad_A + (1 - beta1) * grad_A
                    exp_avg_sq_grad_A = beta2 * exp_avg_sq_grad_A + (1 - beta2) * grad_A ** 2

                    exp_avg_grad_A_hat = exp_avg_grad_A / (1 - beta1 ** step_count)
                    exp_avg_sq_grad_A_hat = exp_avg_sq_grad_A / (1 - beta2 ** step_count)
                    updata_A = s * ((exp_avg_grad_A_hat / (torch.sqrt(exp_avg_sq_grad_A_hat) + eps)) + wd_coef * p)

                    state["exp_avg_grad_A"] = exp_avg_grad_A
                    state["exp_avg_sq_grad_A"] = exp_avg_sq_grad_A

                new_update = updata_TJU + updata_A
                p.add_(new_update, alpha=-current_lr)  # apply gradient update: new_update controls update direction, alpha=-current_lr is the learning rate (step size)

                # Store the current update vector for use in the next iteration
                prev_update.copy_(new_update)  # save current update new_update to prev_update for the next iteration

        # print(f'current s value: {s}')
        return loss