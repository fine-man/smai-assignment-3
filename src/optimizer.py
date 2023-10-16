import numpy as np
from src import *

class Optimizer():
    """
    Optimizer class
    """
    def __init__(
        self,
        model,
        update_rule="sgd",
        optim_config={}):

        self.model = model
        self.update_rule = update_rule
        self.optim_config = optim_config

        # check if the specified update rule exists and replace the string
        # with the update function if it exists
        if not hasattr(optim, self.update_rule):
            raise ValueError(f'Invalid update rule: "{self.update_rule}"')
        self.update_rule = getattr(optim, self.update_rule)

        # reset the config
        self._reset()
    
    def _reset(self):
        """
        Reset the optimizer config
        """
        # copying the default config for each parameter
        self.optim_configs = {}
        for param_name, param in self.model.params.items():
            default = {key:value for key, value in self.optim_config.items()}
            self.optim_configs[param_name] = default
    
    def step(self):
        """
        Weight update step
        """
        for param_name, param in self.model.params.items():
            # getting the weights, gradients and config for this parameter
            w, dw = param, self.model.grads[param_name]
            config = self.optim_configs[param_name]

            # updating the weights/config and setting them
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[param_name] = next_w
            self.optim_configs[param_name] = next_config