import os
import sys

# enter the Foldername here:
FOLDERNAME = "/home/richard/play/IIITH/sem5/smai/assignments/assignment-3-fine-man"

if FOLDERNAME is None or not os.path.exists(FOLDERNAME):
    FOLDERNAME = os.getcwd()

PATHNAME = f"{FOLDERNAME}"
sys.path.append(f"{FOLDERNAME}")

import numpy as np
import pandas as pd
import copy
import os
import argparse
from random import randrange
from PIL import Image
import wandb
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from torch.utils.data import random_split

import torchvision
from torchvision import transforms

from src_torch import *
from src_torch.classifiers import *

def sweep_agent_manager():
    global train_dataset, val_dataset
    run = wandb.init()
    config = re_nest_config(dict(wandb.config))

    # setting the wandb run name for the current config
    model_type = get_model_type(config["model"])
    if model_type == "CNN":
        run_name = make_wandb_run_name_cnn(config)
    elif model_type == "MLP":
        run_name = make_wandb_run_name_mlp(config)
    else:
        return
    print(f"Run Name: {run_name}")
    run.name = run_name

    # start the training
    trigger_training(config, train_dataset, val_dataset)

if __name__ == "__main__":
    # DATA_FOLDER = os.path.join(FOLDERNAME, "SMAI-Dataset-release/IIIT-CFW")
    DATA_FOLDER = os.path.join(FOLDERNAME, "datasets")
    YAML_FOLDER = os.path.join(FOLDERNAME, "yaml-files")
    PERMUTED_MNIST_FILE = os.path.join(DATA_FOLDER, "permuted_mnist.npz")


    # DATA Loading and Pre-Processing
    parser = argparse.ArgumentParser(description="Run Wandb Experiments for Task 5.2 of SMAI Assignment 3")
    
    parser.add_argument(
        '-p', '--path', default=f'{YAML_FOLDER}/task-5.2/permuted-mnist-single-layer.yaml',
        help='Path to the yaml file containing the Wandb sweep configuration')

    args = parser.parse_args()
    path = args.path

    print(DATA_FOLDER, flush=True)
    print(YAML_FOLDER, flush=True)
    print(PERMUTED_MNIST_FILE)

    print(f"Path of sweep configuration file: {path}", flush=True)

    train_dataset, val_dataset, test_dataset = load_and_process_permuted_mnist(PERMUTED_MNIST_FILE)

    print(f"Length of Training Data: {len(train_dataset)}", flush=True)
    print(f"Length of Validation Data: {len(val_dataset)}", flush=True)

    # Reading the wandb config file
    with open(path, "r") as stream:
        try:
            sweep_configuration = yaml.safe_load(stream)
            print(sweep_configuration, flush=True)
        except yaml.YAMLError as exc:
            print(exc, flush=True)
            exit(-1)

    # Starting Wandb
    api_key = "8c09025842609f9e17e5aa0de5aa2ab26314a316"
    wandb.login(key=api_key)
    project_name = sweep_configuration.get("project", "smai-assignment3-task5.2")
    print(f"Wandb Project Name: {project_name}", flush=True)
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
    wandb.agent(sweep_id=sweep_id, function=sweep_agent_manager)