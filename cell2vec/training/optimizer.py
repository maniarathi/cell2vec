class TransformerOptimizer():
    def __init__(self, optimizer, model_dimensionality, warmup_steps=4000):
        self.optimizer = optimizer
        self.model_dimensionality = model_dimensionality
        self.warmup_steps = warmup_steps

        self._step_number = 0

    def step(self):
        self._step_number += 1

        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()

    def rate(self):
        return self.model_dimensionality ** -0.5 * min(self._step_number ** -0.5,
                                                       self._step_number * (self.warmup_steps ** -1.5))
