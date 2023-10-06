import numpy as np
from src import *

"""
General structure of all functions in this file:

def update(w, dw, config=None):
    Input:
        - w: A numpy array giving the current weights
        - dw: A numpy array of the same shape as w giving the gradient of the loss
            with respect to w
        - config: A dictionary containing hyperparameter values such as learning
            rate, momentum, etc. If the update rule requires caching values over many
            iterations, then config will also hold these cached values.
    
    Returns:
        - next_w: The next weight after the update
        - config: The dictionary to be passed to the next iteration of the update rule
"""

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


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent

    config_format:
        - learning_rate: Scalar learning rate
    """
    if config is None:
        config = {}
        config.setdefault("learning_rate", 1e-2)
    
    lr = config["learning_rate"]
    w -= config["learning_rate"] * dw
    return w, config