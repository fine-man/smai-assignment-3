import numpy as np

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
    w -= config["learning_rate"] * dw
    return w, config