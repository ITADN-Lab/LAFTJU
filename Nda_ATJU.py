import torch
from torch.optim.optimizer import Optimizer
import numpy as np
import math


class Nda_ATJU(Optimizer):
    r"""
    修正后的TJU_AdamW优化器，保持原TJU_v3精度的同时实现正确解耦权重衰减

    关键改进点：
    1. 修复权重衰减应用方式：使用当前学习率进行缩放 (current_lr * tju_weight_decay)
    2. 保持原TJU_v3的近似Hessian处理逻辑
    3. 恢复原参数更新顺序，确保数值稳定性
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

        # 参数校验（保持原有严格校验）
        if not 0.0 <= tju_weight_decay:
            raise ValueError(f"Invalid tju_weight_decay: {tju_weight_decay}")
        if weight_decay_type not in ['L2', 'stable', 'AdamW']:  # 修正选项列表
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


            # 定义当前的权重值 s
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
                    raise RuntimeError("TJU_AdamW_Fixed不支持稀疏梯度")

                state = self.state[p]
                # 初始化状态（保持原TJU_v3结构）
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['approx_hessian'] = torch.zeros_like(p)
                    # sgd历史动量值
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # 一阶梯度的指数滑动平均（Adam/W）
                    state['exp_avg_grad_A'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # 二阶梯度的指数滑动平均(AdamW)
                    state["exp_avg_sq_grad_A"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1
                step = state['step']
                self.step_now = step

                # ====== 学习率调度（保持原TJU_v3逻辑） ======
                # current_lr = self._compute_lr(group, step)
                current_lr = tju_lr

                # ====== 核心参数更新 ======
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                approx_hessian = state['approx_hessian']
                momentum_buffer = state["momentum_buffer"]
                exp_avg_grad_A = state['exp_avg_grad_A']
                exp_avg_sq_grad_A = state["exp_avg_sq_grad_A"]

                # 梯度值分配
                grad_TJU = p.grad  # TJU使用的梯度值
                grad_A = p.grad  # A_optim使用的梯度值


                # (1) L2正则化（保持原逻辑）
                if group['weight_decay_type'] == 'L2' and group['tju_weight_decay'] != 0:
                    grad_TJU = grad_TJU.add(p, alpha=group['tju_weight_decay'])

                # (2) 更新动量项（保持原TJU_v3数值稳定性）
                exp_avg.mul_(tju_beta1).add_(grad_TJU, alpha=1 - tju_beta1)
                exp_avg_sq.mul_(tju_beta2).addcmul_(grad_TJU, grad_TJU, value=1 - tju_beta2)

                # (3) 偏置校正（关键！保持原TJU_v3实现）
                bias_corr1 = 1 - tju_beta1 ** step
                bias_corr2 = 1 - tju_beta2 ** step
                step_size = current_lr / bias_corr1  # 合并学习率与一阶偏置校正

                # (4) 近似Hessian处理（保持原TJU_v3的clamp逻辑）
                delta_grad = grad_TJU - (exp_avg / bias_corr1)  # 修正后的梯度变化量

                approx_hessian.mul_(group['beta_h']).addcmul_(
                    delta_grad, delta_grad, value=1 - group['beta_h'])

                if group['rebound'] == 'constant':
                    denom_hessian = approx_hessian.abs().clamp_(min=1e-3)  # 保持原v3的clamp下限
                else:
                    bound_val = max(delta_grad.norm(p=float('inf')).item(), 1e-5)
                    denom_hessian = torch.max(approx_hessian.abs(),
                                              torch.tensor(bound_val, device=p.device))

                # (5) 组合二阶动量（保持原v3的混合逻辑）
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(
                    group['hessian_scale'] * denom_hessian,
                    alpha=1.0
                ).add_(group['tju_eps'])

                # (6) 计算更新方向（关键修改点！恢复原v3的稳定性）
                new_update = exp_avg / denom

                # (7) 处理stable类型权重衰减（保持原v3逻辑）
                if group['weight_decay_type'] == 'stable' and group['tju_weight_decay'] != 0:
                    decay_factor = group['tju_weight_decay'] / denom.mean().clamp(min=1e-8)
                    new_update.add_(p, alpha=decay_factor)

                # ====== AdamW类型权重衰减（关键修正！）====== #
                # 在参数更新时应用解耦衰减（保持与当前学习率无关）
                if group['weight_decay_type'] == 'AdamW' and group['tju_weight_decay'] != 0:
                    p.data.mul_(1 - group['tju_weight_decay'] * current_lr)  # 与学习率解耦的关键修改！

                updata_TJU = (1 - s) * new_update

                if A_optim == 'SGD':
                    # SGD 更新部分
                    grad_A = grad_A.add(p, alpha=a_wd_coef)
                    # 计算参数更新值（sgd）
                    updata_A = s * (grad_A + momentum * momentum_buffer)
                    # 存储动量累计值
                    state["momentum_buffer"] = grad_A + momentum * momentum_buffer

                elif A_optim == 'Adam':
                    # Adam更新部分
                    grad_A = grad_A.add(p, alpha=a_wd_coef)
                    exp_avg_grad_A = a_beta1 * exp_avg_grad_A + (1 - a_beta1) * grad_A
                    exp_avg_sq_grad_A = a_beta2 * exp_avg_sq_grad_A + (1 - a_beta2) * grad_A ** 2

                    exp_avg_grad_A_hat = exp_avg_grad_A / (1 - a_beta1 ** step)
                    exp_avg_sq_grad_A_hat = exp_avg_sq_grad_A / (1 - a_beta2 ** step)
                    updata_A = s * (exp_avg_grad_A_hat / (torch.sqrt(exp_avg_sq_grad_A_hat) + a_eps))

                    state["exp_avg_grad_A"] = exp_avg_grad_A
                    state["exp_avg_sq_grad_A"] = exp_avg_sq_grad_A

                elif A_optim == 'AdamW':
                    # AdamW 更新部分
                    exp_avg_grad_A = a_beta1 * exp_avg_grad_A + (1 - a_beta1) * grad_A
                    exp_avg_sq_grad_A = a_beta2 * exp_avg_sq_grad_A + (1 - a_beta2) * grad_A ** 2

                    exp_avg_grad_A_hat = exp_avg_grad_A / (1 - a_beta1 ** step)
                    exp_avg_sq_grad_A_hat = exp_avg_sq_grad_A / (1 - a_beta2 ** step)
                    updata_A = s * ((exp_avg_grad_A_hat / (torch.sqrt(exp_avg_sq_grad_A_hat) + a_eps)) + a_wd_coef * p)

                    state["exp_avg_grad_A"] = exp_avg_grad_A
                    state["exp_avg_sq_grad_A"] = exp_avg_sq_grad_A


                update = -step_size * updata_TJU -a_lr * updata_A   # 注意：step_size已包含学习率和一阶偏置校正


                # (8) 执行参数更新（保持原v3的更新顺序）
                p.add_(update)

        print(f'当前s值为： {s}')
        return loss

    def _compute_lr(self, group, step):
        """学习率调度（精确保持原TJU_v3实现）"""
        if step <= group['warmup']:
            return group['init_lr'] + (group['base_lr'] - group['init_lr']) * step / group['warmup']

        if not group['use_cosine_scheduler']:
            return group['base_lr']

        t = step - group['warmup']
        T = group['total_steps'] - group['warmup']

        if t <= T:
            return group['base_lr'] * (0.5 * (1 + math.cos(math.pi * t / T)))
        return group['base_lr'] * 0.01  # 保持原v3的后训练阶段学习率