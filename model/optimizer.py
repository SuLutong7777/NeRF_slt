import torch
import logging
import numpy as np

class Create_optimizer():
    def __init__(self, system_param):
        self.opt_type = system_param["opt_type"]
        self.max_lr = system_param["opt_lr_max"]
        self.min_lr = system_param["opt_lr_min"]
        self.decay = system_param["opt_decay"]
        self.num_steps = system_param["num_steps"]
        logging.info('Current Optimizer Type: {}'.format(self.opt_type))

    def get_optimizer(self, model):
        # cycle表示按照三角函数方式循环变换学习率
        # decay表示学习率逐步下降
        if self.opt_type == 'decay':
            # 创建优化器
            # optimizer = torch.optim.RAdam(model.parameters(), lr=max_lr, weight_decay=decay)
            self.optimizer = torch.optim.RAdam(model.parameters(), lr=self.max_lr)
            # 构造指数衰减率
            gamma_cam = (self.min_lr/self.max_lr)**(1./self.num_steps)
            self.lr_fn_main = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma_cam)
        
        elif self.opt_type == 'cycle':
            lr_kwargs = {
                'max_steps': self.num_steps,
                'lr_delay_steps': 1500,
                'lr_delay_mult': 1e-08,
            }
            self.lr_fn_main = lambda step: self.learning_rate_decay(step, lr_init=self.max_lr, lr_final=self.min_lr, **lr_kwargs)
            self.optimizer = torch.optim.RAdam(model.parameters(), lr=self.max_lr, eps=1e-15)
        else:
            print("Unsupport Optimizer!")
            exit()

        return self.optimizer, self.lr_fn_main

    def update_params(self, cur_step):
        if self.opt_type == 'decay':
            self.lr_fn_main.step() 
        elif self.opt_type == 'cycle':
            # 更新学习速率
            learning_rate = self.lr_fn_main(cur_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
        else:
            exit()

    def learning_rate_decay(self, step, lr_init, lr_final, max_steps, lr_delay_steps=0, lr_delay_mult=1):
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.

        return delay_rate * self.log_lerp(step / max_steps, lr_init, lr_final)

    def log_lerp(self, t, v0, v1):
        if v0 <= 0 or v1 <= 0:
            raise ValueError(f'Interpolants {v0} and {v1} must be positive.')
        lv0 = np.log(v0)
        lv1 = np.log(v1)
        return np.exp(np.clip(t, 0, 1) * (lv1 - lv0) + lv0)