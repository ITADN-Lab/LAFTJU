import math


class MyCosineAnnealingLR:
    def __init__(self, optimizer, T_max, tju_lr_min=0, A_lr_min=0, last_epoch=-1):
        """
        Args:
            optimizer (Optimizer): wrapped optimizer
            T_max (int): maximum number of iterations (usually the length of one epoch cycle)
            eta_min (float): minimum learning rate. Default: 0.
            last_epoch (int): index of the last epoch. Default: -1.
        """
        self.optimizer = optimizer
        self.T_max = T_max
        self.tju_lr_min = tju_lr_min
        self.A_lr_min = A_lr_min
        self.last_epoch = last_epoch

        # Key point: we need to record the initial learning rate (base_lr) set at the beginning of the optimizer
        # because subsequent steps will modify the lr in the optimizer, and we need a baseline value to compute from
        self.tju_lrs = [group['tju_lr'] for group in optimizer.param_groups]
        self.A_optim_lrs = [group['a_lr'] for group in optimizer.param_groups]

    def step(self):
        """
        Call this method at the end of each training iteration to update the learning rate
        """
        self.last_epoch += 1

        # Iterate over each parameter group in the optimizer (usually only one group, but iterate for compatibility)
        for param_group, tju_lr in zip(self.optimizer.param_groups, self.tju_lrs):

            # --- Core formula implementation ---
            # cos input range is [0, pi], corresponding to epoch range [0, T_max]
            # when last_epoch = 0, cos(0)=1 -> lr = base_lr
            # when last_epoch = T_max, cos(pi)=-1 -> lr = eta_min

            # To prevent division by zero when T_max is 0 (though it won't be set to 0 in practice)
            if self.T_max == 0:
                step_ratio = 1.0
            else:
                step_ratio = self.last_epoch / self.T_max

            # Compute the cosine factor
            cosine_factor = 0.5 * (1 + math.cos(math.pi * step_ratio))

            # Compute the new learning rate
            new_lr = self.tju_lr_min + (tju_lr - self.tju_lr_min) * cosine_factor

            # Assign the new learning rate to the optimizer
            param_group['tju_lr'] = new_lr


        for param_group, A_optim_lrs in zip(self.optimizer.param_groups, self.A_optim_lrs):

            # Update A_optim parameters

            if self.T_max == 0:
                step_ratio = 1.0
            else:
                step_ratio = self.last_epoch / self.T_max

            # Compute the cosine factor
            cosine_factor = 0.5 * (1 + math.cos(math.pi * step_ratio))

            # Compute the new learning rate
            new_A_lr = self.A_lr_min + (A_optim_lrs - self.A_lr_min) * cosine_factor

            # Assign the new learning rate to the optimizer
            param_group['a_lr'] = new_A_lr



    def get_tju_lr(self):
        """Get the current learning rate, for printing or logging"""
        return [group['tju_lr'] for group in self.optimizer.param_groups]

    def get_A_optim_lr(self):
        """Get the current learning rate, for printing or logging"""
        return [group['a_lr'] for group in self.optimizer.param_groups]