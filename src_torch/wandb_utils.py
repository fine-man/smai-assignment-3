import wandb
import numpy as np
import pandas as pd
from src_torch import *

def re_nest_config(config_dict):
    nested_config = copy.deepcopy(config_dict)
    flattened_params = [key for key in config_dict.keys() if '.' in key]
    for param in flattened_params:
        value = nested_config.pop(param)
        param_levels = param.split('.')
        parent = nested_config
        for i, level in enumerate(param_levels):
            if not isinstance(parent, dict):
                break
            if i + 1 == len(param_levels):
                parent[level] = value
            
            if level not in parent.keys():
                parent[level] = {}
            
            parent = parent[level]

    return nested_config