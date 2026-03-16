import math


class MyCosineAnnealingLR:
    def __init__(self, optimizer, T_max, tju_lr_min=0, A_lr_min=0, last_epoch=-1):
        """
        Args:
            optimizer (Optimizer): 包装的优化器
            T_max (int): 最大的迭代次数 (通常是一个 epoch 周期的长度)
            eta_min (float): 最小学习率. 默认为 0.
            last_epoch (int): 上一个 epoch 的索引. 默认为 -1.
        """
        self.optimizer = optimizer
        self.T_max = T_max
        self.tju_lr_min = tju_lr_min
        self.A_lr_min = A_lr_min
        self.last_epoch = last_epoch

        # 关键点：我们需要记录优化器最开始设置的初始学习率 (base_lr)
        # 因为后续 step 会修改 optimizer 里的 lr，我们需要一个基准值来计算
        self.tju_lrs = [group['tju_lr'] for group in optimizer.param_groups]
        self.A_optim_lrs = [group['a_lr'] for group in optimizer.param_groups]

    def step(self):
        """
        每次训练迭代结束时调用此方法更新学习率
        """
        self.last_epoch += 1

        # 遍历优化器中的每一个参数组 (通常只有一个组，但为了兼容性需遍历)
        for param_group, tju_lr in zip(self.optimizer.param_groups, self.tju_lrs):

            # --- 核心公式实现 ---
            # cos 输入范围是 [0, pi]，对应 epoch 范围 [0, T_max]
            # 当 last_epoch = 0 时，cos(0)=1 -> lr = base_lr
            # 当 last_epoch = T_max 时，cos(pi)=-1 -> lr = eta_min

            # 为了防止 T_max 为 0 导致除以零错误 (虽然实际中不会设为0)
            if self.T_max == 0:
                step_ratio = 1.0
            else:
                step_ratio = self.last_epoch / self.T_max

            # 计算余弦部分
            cosine_factor = 0.5 * (1 + math.cos(math.pi * step_ratio))

            # 计算新的学习率
            new_lr = self.tju_lr_min + (tju_lr - self.tju_lr_min) * cosine_factor

            # 将新学习率赋值给优化器
            param_group['tju_lr'] = new_lr


        for param_group, A_optim_lrs in zip(self.optimizer.param_groups, self.A_optim_lrs):

            # 更新A_optim参数

            if self.T_max == 0:
                step_ratio = 1.0
            else:
                step_ratio = self.last_epoch / self.T_max

            # 计算余弦部分
            cosine_factor = 0.5 * (1 + math.cos(math.pi * step_ratio))

            # 计算新的学习率
            new_A_lr = self.A_lr_min + (A_optim_lrs - self.A_lr_min) * cosine_factor

            # 将新学习率赋值给优化器
            param_group['a_lr'] = new_A_lr



    def get_tju_lr(self):
        """获取当前学习率，用于打印或记录"""
        return [group['tju_lr'] for group in self.optimizer.param_groups]

    def get_A_optim_lr(self):
        """获取当前学习率，用于打印或记录"""
        return [group['a_lr'] for group in self.optimizer.param_groups]