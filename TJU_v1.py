import numpy as np
import torch
from torch.optim.optimizer import Optimizer

class TJU_v1(Optimizer):
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
        beta=0.9,
        eps=1e-4, 
        rebound='constant',
        warmup=500, 
        init_lr=None, 
        weight_decay=0, 
        weight_decay_type=None
    ):
        # -- Parameter validation --
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate value: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta: {beta} (must be in [0.0, 1.0))")
        if rebound not in ['constant', 'belief']:
            raise ValueError(f"Invalid rebound mode: {rebound}, must be 'constant' or 'belief'")
        if not 0 <= warmup:
            raise ValueError(f"Invalid warmup steps: {warmup} (must be >= 0)")

        # If init_lr is not provided, default to init_lr = lr / 1000
        if init_lr is None:
            init_lr = lr / 1000
        if not 0.0 <= init_lr <= lr:
            raise ValueError(f"Invalid init_lr: {init_lr} (must be in [0, lr])")

        # Weight decay coefficient and type
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay} (must be >= 0)")
        if weight_decay_type is None:
            # If rebound='constant', default to L2; otherwise use decoupled
            weight_decay_type = 'L2' if rebound == 'constant' else 'decoupled'
        if weight_decay_type not in ['L2', 'decoupled', 'stable']:
            raise ValueError(f"Invalid weight_decay_type: {weight_decay_type} "
                             "(must be 'L2', 'decoupled', or 'stable')")

        # Store all default hyperparameters in a dictionary
        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            rebound=rebound,
            warmup=warmup,
            init_lr=init_lr,
            base_lr=lr,
            weight_decay=weight_decay,
            weight_decay_type=weight_decay_type
        )
        # TJU_v1 inherits all attributes from the parent class, corresponding to params and defaults
        super(TJU_v1, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(TJU_v1, self).__setstate__(state)

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

        # Iterate over each param group
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            eps = group['eps']
            rebound_mode = group['rebound']
            warmup_steps = group['warmup']
            init_lr = group['init_lr']
            base_lr = group['base_lr']
            wd_coef = group['weight_decay']
            wd_type = group['weight_decay_type']

            # Iterate over parameters in this group
            for p in group['params']:
                if p.grad is None:
                    continue

                # Get the state dictionary for the current parameter
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of first-order gradients
                    state['exp_avg_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
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

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("TJU_v3 does not support sparse gradients.")

                # If using L2 decay, add the decay term directly to the gradient
                if wd_coef != 0 and wd_type == 'L2':
                    grad = grad.add(p, alpha=wd_coef)

                exp_avg_grad = state['exp_avg_grad']
                approx_hessian = state['approx_hessian']
                prev_update = state['prev_update']

                bias_corr = 1 - (beta ** step_count)  
                effective_alpha = (1 - beta) / bias_corr

                # Compute gradient difference
                grad_diff = grad - exp_avg_grad  # Current parameter gradient minus the previously stored gradient

                # Select threshold based on rebound mode
                if rebound_mode == 'belief':
                    rebound_thresh = grad_diff.norm(p=np.inf)   # Dynamically set threshold based on the max absolute value of the current gradient difference (grad_diff)
                else:
                    rebound_thresh = 0.01
                    eps = eps / max(rebound_thresh, 1e-8)  # Prevent division by zero

                # (1) Update first-order gradient EMA: exp_avg_grad ← exp_avg_grad + effective_alpha * grad_diff
                exp_avg_grad.add_(grad_diff, alpha=effective_alpha)

                # (2) Normalize the previous update, used to compute the approximate Hessian update
                prev_update_norm = prev_update.norm(p=4).add(eps)   # Compute L4 norm and add eps to prevent zero denominator
                prev_update.div_(prev_update_norm)  # Normalize the original prev_update by its L4 norm
                prev_update_sq = prev_update.mul(prev_update)   # Compute the element-wise square of the normalized prev_update

                # Compute delta for updating the Hessian
                # dot product of (grad_diff / prev_update_norm) and prev_update
                delta_term = (grad_diff.div_(prev_update_norm)  # Normalize grad_diff
                                       .mul_(prev_update)   # Compute element-wise product of normalized grad_diff and prev_update
                                       .sum()   # Sum over all parameter dimensions to get a scalar
                                       .mul_(-effective_alpha)  # Multiply result by negative effective learning rate effective_alpha
                              ) - approx_hessian.mul(prev_update_sq).sum()  # Multiply approximate Hessian approx_hessian by squared prev_update_sq and sum
                                                                            # Subtract to get delta_term

                # (3) Update Hessian: approx_hessian ← approx_hessian + (delta_term * prev_update_sq)
                approx_hessian.addcmul_(prev_update_sq, delta_term)

                # (4) Compute new update direction new_update
                if rebound_mode == 'belief':
                    denom_h = torch.max(approx_hessian.abs(), rebound_thresh)   # Element-wise maximum of absolute value of approx_hessian and rebound_thresh
                    denom_h.add_(eps / effective_alpha)     # Prevent division by zero
                else:
                    denom_h = approx_hessian.abs().clamp_(min=rebound_thresh)   # Take absolute value of approx_hessian and clamp values below rebound_thresh to rebound_thresh

                new_update = exp_avg_grad.div(denom_h)  # Scale the gradient momentum exp_avg_grad by the inverse curvature to get the final update direction

                # (5) Apply decoupled or stable weight decay if needed
                if wd_coef != 0 and wd_type != 'L2':    # If weight decay is non-zero and type is not L2 regularization
                    if wd_type == 'stable':
                        scaled_decay = wd_coef / max(denom_h.mean().item(), 1e-8)   # Compute mean of denom_h (reflecting parameter curvature).
                                                                                    # Divide wd_coef by this mean to get the scaled decay factor scaled_decay.
                        new_update.add_(p, alpha=scaled_decay)  # Apply the scaled decay factor to parameter p in the update direction new_update
                    else:
                        # decoupled
                        new_update.add_(p, alpha=wd_coef)   # Apply fixed weight decay directly (independent of gradient)

                # (6) Update parameters
                p.add_(new_update, alpha=-current_lr)   # Apply gradient update: new_update controls the update direction, alpha=-current_lr is the learning rate (step size)

                # Store the current update vector for use in the next iteration
                prev_update.copy_(new_update)   # Save the current update new_update to prev_update for use in the next iteration

        return loss