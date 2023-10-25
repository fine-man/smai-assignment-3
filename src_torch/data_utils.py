import numpy as np
import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from torch.utils.data import random_split
import torchvision

from .datasets import DoubleMNIST

# Helper Pre-Processing functions
def convert_to_onehot(label):
    one_hot_label = torch.zeros(10)
    first_digit = label % 10
    one_hot_label[first_digit] = 1.0
    label = label // 10
    if label:
        second_digit = label % 10
        one_hot_label[second_digit] = 1.0
    else:
        one_hot_label[0] = 1.0
    
    return one_hot_label

def onehot_to_number(onehot_label):
    number = 0
    for i, num in enumerate(onehot_label[1:]):
        if num:
            number = number * 10 + num *  (i + 1)
    
    if onehot_label[0]:
        number = number * 10
    
    return int(number)

# Double MNIST Dataset
def load_and_process_double_mnist(path):
    DOUBLE_MNIST_FOLDER = path

    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    test_images = []
    test_labels = []

    forbidden_labels = [11 * i for i in range(10)]

    # Saving path of all images in train set
    for img_folder in glob.glob(os.path.join(DOUBLE_MNIST_FOLDER, "train/*")):
        label = int(os.path.basename(img_folder))
        if label in forbidden_labels:
            continue
        for img_file in glob.glob(os.path.join(img_folder, "*")):
            image = torchvision.io.read_image(img_file)
            train_labels.append(label)
            train_images.append(image)

    # Saving path of all images in val set
    for img_folder in glob.glob(os.path.join(DOUBLE_MNIST_FOLDER, "val/*")):
        label = int(os.path.basename(img_folder))
        if label in forbidden_labels:
            continue
        for img_file in glob.glob(os.path.join(img_folder, "*")):
            image = torchvision.io.read_image(img_file)
            val_labels.append(label)
            val_images.append(image)

    # Saving path of all images in test set
    for img_folder in glob.glob(os.path.join(DOUBLE_MNIST_FOLDER, "test/*")):
        label = int(os.path.basename(img_folder))
        if label in forbidden_labels:
            continue
        for img_file in glob.glob(os.path.join(img_folder, "*")):
            image = torchvision.io.read_image(img_file)
            test_labels.append(label)
            test_images.append(image)

    train_images = torch.stack(train_images, dim=0)
    val_images = torch.stack(val_images, dim=0)
    test_images = torch.stack(test_images, dim=0)

    train_dataset = DoubleMNIST(train_images, train_labels, target_transform=convert_to_onehot)
    val_dataset = DoubleMNIST(val_images, val_labels, target_transform=convert_to_onehot)
    test_dataset = DoubleMNIST(test_images, test_labels, target_transform=convert_to_onehot)

    return train_dataset, val_dataset, test_dataset

# Permuted MNIST Dataset
def load_and_process_permuted_mnist(path):
    # loading the permuted mnist data
    data = np.load(path)

    # creating and Tensor datasets
    train_images = torch.tensor(data['train_images'])/255.0
    train_images = train_images.unsqueeze(1)
    train_labels = torch.tensor(data['train_labels'])
    test_images = torch.tensor(data['test_images'])/255.0
    test_images = test_images.unsqueeze(1)
    test_labels = torch.tensor(data['test_labels'])

    complete_train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    
    # Splitting into Train and Val sets
    val_ratio = 0.3
    val_size = int(val_ratio * len(complete_train_dataset))
    train_size = len(complete_train_dataset) - val_size
    train_dataset, val_dataset = random_split(complete_train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset