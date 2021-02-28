from asteroid.engine.schedulers import *
import numpy as np


class ExponentialWarmup(BaseScheduler):
    def __init__(self, optimizer, max_lr, rampup_length, exponent=-5.0):
        super().__init__(optimizer)
        self.rampup_len = rampup_length
        self.max_lr = max_lr
        self.step_num = 1
        self.exponent = exponent

    def _get_scaling_factor(self):

        if self.rampup_len == 0:
            return 1.0
        else:

            current = np.clip(self.step_num, 0.0, self.rampup_len)
            phase = 1.0 - current / self.rampup_len
            return float(np.exp(self.exponent * phase * phase))

    def _get_lr(self):
        return self.max_lr * self._get_scaling_factor()
