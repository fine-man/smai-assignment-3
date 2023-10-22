import torch
import torch.nn as nn

# Define a Simple CNN architecture for MNIST
class SimpleCNN(nn.Module):
    def __init__(
        self,
        num_classes=10,
        kernel_size=3,
        stride=1,
        num_channels=16,
        dropout=0.25):
        # initilizaing parent class
        super(SimpleCNN, self).__init__()

        # First Conv block
        in_dim = 28 # Mnist images are 28*28
        k = kernel_size
        s = stride
        pad = (in_dim * (s - 1) + k - s)//2

        self.conv1 = nn.Conv2d(
            1, num_channels, kernel_size=k, stride=s, padding=pad
        )
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Conv block
        in_dim = 14
        in_channels = num_channels
        out_channels = 2 * num_channels
        k = kernel_size
        s = stride
        pad = (in_dim * (s - 1) + k - s)//2

        self.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=k, stride=s, padding=pad
        )
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # dropout layer
        self.dropout = nn.Dropout(p=dropout)
        # Flat fully connected layer
        self.fc = nn.Linear(out_channels * 7 * 7, num_classes)  # MNIST images are 28x28

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x