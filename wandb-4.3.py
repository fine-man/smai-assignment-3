import os
import sys

# enter the Foldername here:
FOLDERNAME = "/home/richard/play/IIITH/sem5/smai/assignments/assignment-3-fine-man"

if FOLDERNAME is None or not os.path.exists(FOLDERNAME):
    FOLDERNAME = os.getcwd()

PATHNAME = f"{FOLDERNAME}"
sys.path.append(f"{FOLDERNAME}")

# DATA_FOLDER = os.path.join(FOLDERNAME, "SMAI-Dataset-release/IIIT-CFW")
DATA_FOLDER = os.path.join(FOLDERNAME, "datasets")
YAML_FOLDER = os.path.join(FOLDERNAME, "yaml-files")
print(DATA_FOLDER)
print(YAML_FOLDER)

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

def get_model(config):
    num_classes = config.pop("num_classes", 10)
    num_channels = config.pop("num_channels", 16)
    dropout = config.pop("dropout", 0.1)
    num_conv_layers = config.pop("num_conv_layers", 2)
    kernel_size = config.pop("kernel_size", 3)
    stride = config.pop("stride", 1)
    
    model = SimpleCNN(
        num_classes, kernel_size, stride, num_channels, dropout
    )
    return model

def get_criterion(crit_name):
    if crit_name == "CE":
        return nn.CrossEntropyLoss()
    else:
        return nn.CrossEntropyLoss()

def get_optimizer(config, model):
    lr = config["learning_rate"]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer

def trigger_training(config, train_dataset, val_dataset):
    np.random.seed(42)
    # getting the model, criterion and optimizer
    model = get_model(config["model"])
    print(model)
    criterion = get_criterion(config["criterion"])
    optimizer = get_optimizer(config["optimizer"], model)

    # training config
    train_config = config["training"]
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    train(
        model, criterion, optimizer, train_dataset, val_dataset, device=device, **train_config)
    
    return model

def make_wandb_run_name(config):
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

def sweep_agent_manager():
    global train_dataset, val_dataset
    run = wandb.init()
    config = re_nest_config(dict(wandb.config))
    # setting the wandb run name for the current config
    run_name = make_wandb_run_name(config)
    print(f"\nRun Name: {run_name}\n")
    run.name = run_name
    # start the training
    trigger_training(config, train_dataset, val_dataset)

if __name__ == "__main__":
    # DATA Loading and Pre-Processing
    parser = argparse.ArgumentParser(description="Run Wandb Experiments for Task 4.3 of SMAI Assignment 3")
    
    parser.add_argument(
        '-p', '--path', default=f'{YAML_FOLDER}/task-4.3/two-conv-grid.yaml',
        help='Path to the yaml file containing the Wandb sweep configuration')

    args = parser.parse_args()
    path = args.path
    print(f"Path of sweep configuration file: {path}")

    # Loading the train/val/test data
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # loading the MNIST data
    mnist_train = torchvision.datasets.MNIST(
        root=DATA_FOLDER, train=True,
        transform=transform, download=True
    )

    # Splitting into Train/Val splits
    val_ratio = 0.3
    val_size = int(val_ratio * len(mnist_train))
    train_size = len(mnist_train) - val_size

    train_dataset, val_dataset = random_split(mnist_train, [train_size, val_size])

    print(f"Length of Training Data: {len(train_dataset)}")
    print(f"Length of Validation Data: {len(val_dataset)}")

    # Reading the wandb config file
    with open(path, "r") as stream:
        try:
            sweep_configuration = yaml.safe_load(stream)
            print(sweep_configuration)
        except yaml.YAMLError as exc:
            print(exc)
            exit(-1)

    # Starting Wandb
    api_key = "8c09025842609f9e17e5aa0de5aa2ab26314a316"
    wandb.login(key=api_key)
    project_name = sweep_configuration.get("project", "smai-assignment3-task4")
    print(f"Wandb Project Name: {project_name}")
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
    wandb.agent(sweep_id=sweep_id, function=sweep_agent_manager)