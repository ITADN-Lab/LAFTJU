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
        tju_lr (float):
            Base learning rate for updates.
        tju_beta (float, optional):
            Momentum factor for gradient EMA. Default: 0.9
        tju_eps (float, optional):
            A small constant for numerical stability (denominator addition). Default: 1e-4
        rebound (str, optional):
            Mode for bounding the diagonal Hessian. {'constant', 'belief'}. Default: 'constant'
        warmup (int, optional):
            Number of warmup steps during which the learning rate ramps from init_lr to tju_lr. Default: 500
        init_lr (float, optional):
            Learning rate used at start of warmup. Default: tju_lr/1000
        tju_weight_decay (float, optional):
            Weight decay coefficient. Default: 0
        weight_decay_type (str, optional):
            Weight decay type: {'L2', 'decoupled', 'stable'}.
            If unset, defaults to 'L2' for 'constant' rebound or 'decoupled' otherwise.
    """

    def __init__(
            self,
            params,
            tju_lr,
            a_lr,
            tju_beta1=0.9,
            tju_beta2=0.999,
            a_beta1=0.9,
            a_beta2=0.999,
            tju_eps=1e-8,
            a_eps=1e-8,
            momentum=0,
            rebound='constant',
            warmup=20,
            init_lr=None,
            tju_weight_decay=0,
            a_weight_decay=0,
            weight_decay_type=None,
            A_optim=None,
            total_epoch=0,
            epoch_now=0
    ):

        self.epoch_now = epoch_now
        self.total_epoch = total_epoch
        self.step_now = 0
        # -- 参数合法性检查 --
        if not 0.0 < tju_lr:
            raise ValueError(f"Invalid learning rate value: {tju_lr}")
        if not 0.0 < a_lr:
            raise ValueError(f"Invalid learning rate value: {a_lr}")
        if not 0.0 <= tju_eps:
            raise ValueError(f"Invalid epsilon value: {tju_eps}")
        if not 0.0 <= a_eps:
            raise ValueError(f"Invalid epsilon value: {a_eps}")
        if not 0.0 <= tju_beta1 < 1.0:
            raise ValueError(f"Invalid tju_beta1: {tju_beta1} (must be in [0.0, 1.0))")
        if not 0.0 <= tju_beta2 < 1.0:
            raise ValueError(f"Invalid beta1: {tju_beta2} (must be in [0.0, 1.0))")
        if not 0.0 <= a_beta1 < 1.0:
            raise ValueError(f"Invalid a_beta1: {a_beta1} (must be in [0.0, 1.0))")
        if not 0.0 <= a_beta2 < 1.0:
            raise ValueError(f"Invalid a_beta2: {a_beta2} (must be in [0.0, 1.0))")
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

        # 若未给出 init_lr，则默认 init_lr = tju_lr / 1000
        if init_lr is None:
            init_lr = tju_lr / 1000
        if not 0.0 <= init_lr <= tju_lr:
            raise ValueError(f"Invalid init_lr: {init_lr} (must be in [0, tju_lr])")

        # 权重衰减系数与方式
        if tju_weight_decay < 0.0:
            raise ValueError(f"Invalid tju_weight_decay: {tju_weight_decay} (must be >= 0)")
        if a_weight_decay < 0.0:
            raise ValueError(f"Invalid a_weight_decay: {a_weight_decay} (must be >= 0)")
        if weight_decay_type is None:
            # 如果 rebound='constant' 则缺省用 L2，否则用 decoupled
            weight_decay_type = 'L2' if rebound == 'constant' else 'decoupled'
        if weight_decay_type not in ['L2', 'decoupled', 'stable']:
            raise ValueError(f"Invalid weight_decay_type: {weight_decay_type} "
                             "(must be 'L2', 'decoupled', or 'stable')")
        if A_optim not in ['SGD', 'Adamm', 'AdamW', None]:
            raise ValueError(f"Invalid A_optim: {A_optim} "
                             "(must be 'SGD', 'Adam', or 'AdamW')")

        # 使用字典统一存储默认超参数
        defaults = dict(
            tju_lr=tju_lr,
            a_lr=a_lr,
            tju_beta1=tju_beta1,
            tju_beta2=tju_beta2,
            a_beta1=a_beta1,
            a_beta2=a_beta2,
            tju_eps=tju_eps,
            a_eps=a_eps,
            momentum=momentum,
            rebound=rebound,
            warmup=warmup,
            init_lr=init_lr,
            base_lr=tju_lr,
            tju_weight_decay=tju_weight_decay,
            a_weight_decay=a_weight_decay,
            weight_decay_type=weight_decay_type,
            A_optim=A_optim,
        )
        # ATJU 继承父类的所有属性，分别对应父类 params, defaults
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

        # 依次遍历各个 param group
        for group in self.param_groups:
            tju_lr = group['tju_lr']
            a_lr = group["a_lr"]
            tju_beta1 = group['tju_beta1']
            tju_beta2 = group['tju_beta2']
            a_beta1 = group['a_beta1']
            a_beta2 = group['a_beta2']
            tju_eps = group['tju_eps']
            a_eps = group['a_eps']
            momentum = group["momentum"]
            rebound_mode = group['rebound']
            warmup_steps = group['warmup']
            init_lr = group['init_lr']
            base_lr = group['base_lr']
            wd_coef = group['tju_weight_decay']
            a_wd_coef = group["a_weight_decay"]
            wd_type = group['weight_decay_type']
            A_optim = group['A_optim']

            # 定义当前的权重值 s
            if self.epoch_now == 1:
                self.total_epoch = self.total_epoch * (self.step_now + 1)
                self.epoch_now += 1

            if self.epoch_now >= 1:
                # s = self.step_now / self.total_epoch
                s = math.tanh((self.step_now / self.total_epoch)*2)
            else:
                s = 0

            # 遍历本组中的参数
            for p in group['params']:
                if p.grad is None:
                    continue

                # 取当前参数的 state 字典
                state = self.state[p]

                # 初始化 state
                if len(state) == 0:
                    state['step'] = 0
                    # sgd历史动量值
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # 一阶梯度的指数滑动平均(TJU)
                    state['exp_avg_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # 一阶梯度的指数滑动平均（Adam/W）
                    state['exp_avg_grad_A'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # 二阶梯度的指数滑动平均(AdamW)
                    state["exp_avg_sq_grad_A"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # 近似 Hessian 的指数滑动平均
                    state['approx_hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # 上一次更新方向
                    state['prev_update'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                step_count = state['step']
                if step_count < warmup_steps:
                    # 线性从 init_lr → base_lr
                    current_lr = (base_lr - init_lr) * (step_count / warmup_steps) + init_lr
                else:
                    current_lr = tju_lr

                step_count += 1
                state['step'] = step_count
                self.step_now = step_count

                grad = p.grad  # 原始梯度值
                grad_TJU = p.grad  # TJU使用的梯度值
                grad_A = p.grad  # A_optim使用的梯度值

                if grad.is_sparse:
                    raise RuntimeError("ATJU does not support sparse gradients.")

                #   TJU更新部分
                # 如果是 L2 衰减，直接将衰减项加到梯度
                if wd_coef != 0 and wd_type == 'L2':
                    grad_TJU = grad_TJU.add(p, alpha=wd_coef)

                exp_avg_grad = state['exp_avg_grad']
                momentum_buffer = state["momentum_buffer"]
                exp_avg_grad_A = state['exp_avg_grad_A']
                exp_avg_sq_grad_A = state["exp_avg_sq_grad_A"]
                approx_hessian = state['approx_hessian']
                prev_update = state['prev_update']

                bias_corr = 1 - (tju_beta1 ** step_count)
                effective_alpha = (1 - tju_beta1) / bias_corr

                # 计算梯度差
                grad_diff = grad_TJU - exp_avg_grad  # 当前 参数梯度 减去 过往 存储梯度

                # 根据 rebound 模式选择阈值
                if rebound_mode == 'belief':
                    rebound_thresh = grad_diff.norm(p=np.inf)  # 根据当前梯度差（grad_diff） 的最大绝对值 动态设定阈值
                else:
                    rebound_thresh = 0.01
                    tju_eps = tju_eps / max(rebound_thresh, 1e-8)  # 防止除以 0

                # (1) 更新一阶梯度平均: exp_avg_grad ← exp_avg_grad + effective_alpha * grad_diff
                exp_avg_grad.add_(grad_diff, alpha=effective_alpha)

                # (2) 归一化上一次更新，用于计算近似 Hessian 的更新
                prev_update_norm = prev_update.norm(p=4).add(tju_eps)  # 计算L4范数，并加上eps防止分母为0
                prev_update.div_(prev_update_norm)  # 对原 prev_update 依据L4范数进行归一化
                prev_update_sq = prev_update.mul(prev_update)  # 计算上述归一化后的 prev_update 每个项的平方值

                # 计算 delta，用于更新 Hessian
                # grad_diff / prev_update_norm 与 prev_update 做内积
                delta_term = (grad_diff.div_(prev_update_norm)  # 将 grad_diff 归一化
                              .mul_(prev_update)  # 计算归一化后的 梯度差grad_diff 与 历史更新量prev_update 的点积
                              .sum()  # 对所有参数维度求和，得到标量值
                              .mul_(-effective_alpha)  # 将结果 乘以 负的有效学习率 effective_alpha
                              ) - approx_hessian.mul(
                    prev_update_sq).sum()  # 将近似的 Hessian矩阵 approx_hessian 与历史更新量的平方 prev_update_sq 相乘后求和
                # 二者作差得 delta_term

                # (3) 更新 Hessian: approx_hessian ← approx_hessian + (delta_term * prev_update_sq)
                approx_hessian.addcmul_(prev_update_sq, delta_term)

                # (4) 计算新的更新方向 new_update
                if rebound_mode == 'belief':
                    denom_h = torch.max(approx_hessian.abs(),
                                        rebound_thresh)  # 取 approx_hessian 的绝对值与 rebound_thresh 的逐元素最大值
                    denom_h.add_(tju_eps / effective_alpha)  # 防止除0
                else:
                    denom_h = approx_hessian.abs().clamp_(
                        min=rebound_thresh)  # 取 approx_hessian 的绝对值，将小于 rebound_thresh 的值截断为 rebound_thresh

                new_update = exp_avg_grad.div(denom_h)  # 将历史梯度动量 exp_avg_grad 按曲率倒数缩放，得到最终更新方向

                # (5) 如果需要 decoupled 或 stable wd
                if wd_coef != 0 and wd_type != 'L2':  # 如果权重衰退不为0， 且方式不为L2正则化
                    if wd_type == 'stable':
                        scaled_decay = wd_coef / max(denom_h.mean().item(), 1e-8)  # 计算 denom_h 的均值（反映参数曲率）。
                        # 用 wd_coef 除以该均值，得到缩放后的衰减因子 scaled_decay。
                        new_update.add_(p, alpha=scaled_decay)  # 将缩放后的衰减因子作用于参数 p，更新方向 new_update
                    else:
                        # decoupled
                        new_update.add_(p, alpha=wd_coef)  # 直接应用固定权重的衰减（与梯度无关）

                # (6) 计算参数更新值(TJU)
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

                    exp_avg_grad_A_hat = exp_avg_grad_A / (1 - a_beta1 ** step_count)
                    exp_avg_sq_grad_A_hat = exp_avg_sq_grad_A / (1 - a_beta2 ** step_count)
                    updata_A = s * (exp_avg_grad_A_hat / (torch.sqrt(exp_avg_sq_grad_A_hat) + a_eps))

                    state["exp_avg_grad_A"] = exp_avg_grad_A
                    state["exp_avg_sq_grad_A"] = exp_avg_sq_grad_A

                elif A_optim == 'AdamW':
                    # AdamW 更新部分
                    exp_avg_grad_A = a_beta1 * exp_avg_grad_A + (1 - a_beta1) * grad_A
                    exp_avg_sq_grad_A = a_beta2 * exp_avg_sq_grad_A + (1 - a_beta2) * grad_A ** 2

                    exp_avg_grad_A_hat = exp_avg_grad_A / (1 - a_beta1 ** step_count)
                    exp_avg_sq_grad_A_hat = exp_avg_sq_grad_A / (1 - a_beta2 ** step_count)
                    updata_A = s * ((exp_avg_grad_A_hat / (torch.sqrt(exp_avg_sq_grad_A_hat) + a_eps)) + a_wd_coef * p)

                    state["exp_avg_grad_A"] = exp_avg_grad_A
                    state["exp_avg_sq_grad_A"] = exp_avg_sq_grad_A

                new_update = -current_lr * updata_TJU - a_lr * updata_A
                p.add_(new_update)  # 进行梯度更新，new_update控制更新方向

                # 存放本次的更新向量，供下次计算使用
                prev_update.copy_(new_update)  # 将当前更新量 new_update 保存到 prev_update 中，供下次迭代使用

        print(f'当前s值为： {s}')
        return loss