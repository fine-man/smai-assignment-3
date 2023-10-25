import wandb
import numpy as np
import pandas as pd
from src_torch import *
import copy

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

def make_wandb_run_name_mlp(config):
    model_config = config["model"]
    optim_config = config["optimizer"]
    train_config = config["training"]

    num_layers = model_config.get("num_layers", 1)
    
    hidden_dims = []
    for i in range(1, num_layers + 1):
        dim = model_config[f"hidden_dims{i}"]
        hidden_dims.append(str(dim))
    
    h = "-".join(hidden_dims)

    lr = optim_config["learning_rate"]
    batch = train_config["batch_size"]

    run_name = f"H{h}-lr{lr}-batch{batch}"
    return run_name

def make_wandb_run_name_cnn(config):
    model_config = config["model"]
    optim_config = config["optimizer"]
    train_config = config["training"]

    channels = model_config["num_channels"]
    kernel = model_config["kernel_size"]
    stride = model_config["stride"]
    dropout = model_config["dropout"]

    lr = optim_config["learning_rate"]
    batch = train_config["batch_size"]

    run_name = f"C{channels}-k{kernel}-s{stride}-dropout{dropout}-lr{lr}-batch{batch}"
    return run_name