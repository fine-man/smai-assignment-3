import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from torch.utils.data import random_split

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