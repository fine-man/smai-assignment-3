import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .transforms import AddGaussianNoise

class NoisyMNIST(Dataset):
    def __init__(self, mnist_dataset, mean=0, std=1, autoencoder=False, transform=None, target_transform=None):
        self.mnist = mnist_dataset
        self.mean = 0
        self.std = 1

        self.autoencoder = autoencoder
        self.transform = transform
        self.target_transform = target_transform
        self.add_gaussian_noise = AddGaussianNoise(mean, std)
    
    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        # Adding Gaussian noise to the image
        noisy_image = self.add_gaussian_noise(image)
        noisy_image = torch.clip(noisy_image, 0, 1)

        if self.autoencoder is False:
            return noisy_image, label
        else:
            return noisy_image, image