import torch
from torch.optim.optimizer import Optimizer
import numpy as np


class TJU_v3(Optimizer):
    r"""
    adamWa: An improved variant integrating Adam-like updates and approximate Hessian.

    Changes/Notes:
      1) The approximate Hessian is scaled and added to Adam's second moment rather
         than multiplied directly, preventing overly small steps.
      2) Rebound clamp adjusted to smaller default for 'constant' mode,
         reducing the chance of vanishing updates.
      3) Warmpup steps decreased to accelerate early learning.
      4) beta_h default set to 0.85.

    Args:
        params (iterable):
            Model parameters to optimize (can be a single iterable or multiple param groups).
        lr (float):
            Base learning rate for updates.
        betas (tuple(float, float), optional):
            Adam-like coefficients for computing running averages of gradient (beta1)
            and its square (beta2). Default: (0.9, 0.999).
        beta_h (float, optional):
            Decay factor for approximate Hessian. Default: 0.85
        eps (float, optional):
            A small constant for numerical stability in denominators. Default: 1e-8
        rebound (str, optional):
            Mode for bounding the diagonal Hessian: {'constant', 'belief'}. Default: 'constant'
        warmup (int, optional):
            Number of warmup steps during which LR linearly ramps from init_lr to lr. Default: 100
        init_lr (float, optional):
            Learning rate at the start of warmup. Default: lr / 1000
        weight_decay (float, optional):
            Weight decay coefficient. Default: 0
        weight_decay_type (str, optional):
            Weight decay type: {'L2', 'decoupled', 'stable'}. Default: 'L2'
        hessian_scale (float, optional):
            Scaling factor for approximate Hessian contribution. Default: 0.1
    """
    r"""
    TJU_v3_improved: 在 adamWa 基础上增添三大改进：
    1）在内部引入简单的余弦退火学习率 (可选)，使后期学习率可继续下降。
    2）将 hessian_scale 缩小到 0.05，减弱近似 Hessian 对二阶动量的过大放大。
    3）提升 constant 模式下的 clamp 下限（由 1e-4 提升为 1e-3），避免更新步幅过度缩小。如果不需要内部的学习率调度，可将 use_cosine_scheduler = False。  
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        beta_h=0.85,
        eps=1e-8,
        rebound='constant',
        warmup=100,
        init_lr=None,
        weight_decay=0.0,
        weight_decay_type='L2',
        hessian_scale=0.05,          # 将默认 hessian_scale 从 0.1 下调为 0.05
        total_steps=10000,           # 训练总步数(用于余弦退火)
        use_cosine_scheduler=True    # 是否启用简单的余弦退火
    ):
        # 参数合法性检查
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps {eps}")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 {beta2}")
        if not 0.0 <= beta_h < 1.0:
            raise ValueError(f"Invalid beta_h {beta_h}")
        if rebound not in ['constant', 'belief']:
            raise ValueError(f"Invalid rebound mode {rebound}, must be 'constant' or 'belief'")
        if warmup < 0:
            raise ValueError(f"Invalid warmup steps {warmup}")
        if init_lr is None:
            init_lr = lr / 1000.0
        if not 0.0 <= init_lr <= lr:
            raise ValueError(f"Invalid init_lr {init_lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay {weight_decay}")
        if weight_decay_type not in ['L2', 'decoupled', 'stable']:
            raise ValueError(f"Invalid weight_decay_type {weight_decay_type}")
        if not 0.0 <= hessian_scale <= 1.0:
            raise ValueError(f"Invalid hessian_scale {hessian_scale} (must be in [0,1])")
        if total_steps <= 0:
            raise ValueError(f"Invalid total_steps {total_steps}, must be > 0.")

        defaults = dict(
            lr=lr,
            betas=betas,
            beta_h=beta_h,
            eps=eps,
            rebound=rebound,
            warmup=warmup,
            init_lr=init_lr,
            base_lr=lr,
            weight_decay=weight_decay,
            weight_decay_type=weight_decay_type,
            hessian_scale=hessian_scale,
            total_steps=total_steps,
            use_cosine_scheduler=use_cosine_scheduler
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional):
                A closure that re-evaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历所有参数组
        for group in self.param_groups:
            # 取出参数组里的一些超参
            lr = group['lr']
            beta1, beta2 = group['betas']
            beta_h = group['beta_h']
            eps = group['eps']
            rebound_mode = group['rebound']
            warmup_steps = group['warmup']
            init_lr = group['init_lr']
            base_lr = group['base_lr']
            weight_decay = group['weight_decay']
            weight_decay_type = group['weight_decay_type']
            hessian_scale = group['hessian_scale']
            total_steps = group['total_steps']
            use_cosine_scheduler = group['use_cosine_scheduler']

            # 对每个参数进行更新
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("TJU_v3_improved does not support sparse gradients")

                # 获取参数的状态字典
                state = self.state[p]
                if len(state) == 0:
                    # 初始化状态
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)  # Adam 一阶动量
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)  # Adam 二阶动量
                    state['approx_hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)  # 近似 Hessian

                step_i = state['step']
                step_i += 1
                state['step'] = step_i

                # ====== Warmup 阶段学习率 (线性从 init_lr 到 base_lr) ======
                if step_i <= warmup_steps:
                    current_lr = (base_lr - init_lr) * (step_i / warmup_steps) + init_lr
                else:
                    current_lr = base_lr

                # ====== 可选的余弦退火学习率 (后期衰减) ======
                if use_cosine_scheduler and step_i > warmup_steps:
                    # 计算在 warmup 结束之后已进行的步数
                    t = step_i - warmup_steps
                    T = total_steps - warmup_steps
                    # 简单余弦退火策略
                    if t <= T:
                        # factor 从 1 (t=0) 降到 0 (t=T)
                        alpha = 0.5 * (1.0 + np.cos(np.pi * t / T))
                        current_lr = base_lr * alpha
                    else:
                        # 训练到 total_steps 之后，可以保持更小学习率
                        current_lr = base_lr * 0.01

                # Adam 状态
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                approx_hessian = state['approx_hessian']

                #=========== (1) Weight Decay: L2 ===========
                if weight_decay != 0 and weight_decay_type == 'L2':
                    grad = grad.add(p, alpha=weight_decay)

                #=========== (2) 更新 Adam 的一阶、二阶动量 ===========
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                #=========== (3) 偏置修正 ===========
                bias_correction1 = 1 - beta1 ** step_i
                bias_correction2 = 1 - beta2 ** step_i

                corrected_first = exp_avg.div(bias_correction1)
                corrected_second = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)

                #=========== (4) 近似 Hessian 更新 ===========
                # delta_grad 可以选用 (grad - corrected_first) 或 (grad - exp_avg)
                delta_grad = grad - corrected_first
                approx_hessian.mul_(beta_h).addcmul_(delta_grad, delta_grad, value=1 - beta_h)

                # rebound_mode = 'constant' or 'belief'
                # constant 模式下，我们提高 clamp_(min=1e-3)
                if rebound_mode == 'constant':
                    denom_hessian = approx_hessian.abs().clamp_(min=1e-3)
                else:
                    # belief 模式可以根据无穷范数调整，避免过小
                    bound_val = max(delta_grad.norm(p=float('inf')).item(), 1e-5)
                    denom_hessian = torch.max(approx_hessian.abs(), torch.tensor(bound_val, device=p.device))

                # 将 Hessian 以加法的方式融合到 Adam 的二阶动量上
                combined_denom = corrected_second + hessian_scale * denom_hessian
                combined_denom.add_(eps)

                #=========== (5) 求最终更新方向 ===========
                update_dir = corrected_first.div(combined_denom)

                #=========== (6) 其余 weight decay (decoupled / stable) ===========
                if weight_decay != 0 and weight_decay_type != 'L2':
                    if weight_decay_type == 'stable':
                        # 根据 combined_denom 均值进行 scale
                        scaled_decay = weight_decay / max(combined_denom.mean().item(), 1e-8)
                        update_dir.add_(p, alpha=scaled_decay)
                    else:
                        # decoupled
                        update_dir.add_(p, alpha=weight_decay)

                #=========== (7) 更新参数 ===========
                p.add_(update_dir, alpha=-current_lr)

        return loss