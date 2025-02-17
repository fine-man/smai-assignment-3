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
    next_w = w - config["learning_rate"] * dw
    return next_w, config

def adam(w, dw, config=None):
    """
    Performs Adam update rule with bias correction

   config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number
    """
    if config is None:
        config = {}

    config.setdefault("learning_rate", 1e-2)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)
    
    next_w = None

    # Adam hypterparameters
    m, v, t = config["m"], config["v"], config["t"]
    beta1, beta2 = config["beta1"], config["beta2"]
    lr, epsilon = config["learning_rate"], config["epsilon"]

    # Adam update equations
    t += 1
    m = beta1 * m + (1 - beta1) * dw
    mt = m / (1 - beta1**t)
    v = beta2 * v + (1 - beta2) * (dw**2)
    vt = v / (1 - beta2**t)

    next_w = w - lr * mt/np.sqrt(vt + epsilon)

    # updating the config
    config["m"], config["v"], config["t"] = m, v, t
    return next_w, config