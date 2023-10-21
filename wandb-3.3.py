import os
import sys

# enter the Foldername here:
FOLDERNAME = "/home/richard/play/IIITH/sem5/smai/assignments/assignment-3-fine-man"

if FOLDERNAME is None or not os.path.exists(FOLDERNAME):
    FOLDERNAME = os.getcwd()

PATHNAME = f"{FOLDERNAME}"
sys.path.append(f"{FOLDERNAME}")

DATA_FOLDER = os.path.join(FOLDERNAME, "datasets")
YAML_FOLDER = os.path.join(FOLDERNAME, "yaml-files")

if not os.path.exists(DATA_FOLDER):
    print(f"File {DATA_FOLDER} doesn't exists")
if not os.path.exists(YAML_FOLDER):
    print(f"File {YAML_FOLDER} doesn't exists")

print(f"Path of Data folder: {DATA_FOLDER}")
print(f"Path of YAML folder: {YAML_FOLDER}")

# IMPORTS
import numpy as np
import pandas as pd
import copy
import os
import argparse
from random import randrange

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import wandb
import yaml

from src import *
from src.classifiers import *

def sweep_agent_manager():
    global X_train, y_train, X_val, y_val
    run = wandb.init()
    config = re_nest_config(dict(wandb.config))
    # setting the wandb run name for the current config
    run_name = make_wandb_run_name(config)
    print(f"Run Name: {run_name}")
    run.name = run_name
    # start the training
    trigger_training(config, X_train, y_train, X_val, y_val)

if __name__ == "__main__":
    # DATA Loading and Pre-Processing
    parser = argparse.ArgumentParser(description="Run Wandb Experiments for Task 3.3 of SMAI Assignment 3")
    
    parser.add_argument(
        '-p', '--path', default=f'{YAML_FOLDER}/task-3.3/single-layer-grid-search.yaml',
        help='Path to the yaml file containing the Wandb sweep configuration')

    args = parser.parse_args()
    path = args.path
    print(f"Path of sweep configuration file: {path}")

    # Loading the train/val/test data
    housing_data_path = os.path.join(DATA_FOLDER, "housingdata.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = get_and_process_housing_dataset(housing_data_path)

    print(f"Size of training set: {X_train.shape}")
    print(f"Size of validation set: {X_val.shape}")
    print(f"Size of testing set: {X_test.shape}")

    # Reading the wandb config file
    with open(path, "r") as stream:
        try:
            sweep_configuration = yaml.safe_load(stream)
            print(sweep_configuration)
        except yaml.YAMLError as exc:
            print(exc)
            exit(-1)

    # Starting Wandb
    wandb.login()
    project_name = sweep_configuration.get("project", "smai-assignment3-task3")
    print(f"Wandb Project Name: {project_name}")
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
    wandb.agent(sweep_id=sweep_id, function=sweep_agent_manager)