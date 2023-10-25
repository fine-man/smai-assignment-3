from ..model_utils import get_activation_function
import torch
import torch.nn as nn

# Define a Simple CNN architecture for MNIST
class SimpleCNN(nn.Module):
    def __init__(
        self,
        input_dim=28,
        num_classes=10,
        kernel_size=3,
        stride=1,
        num_channels=16,
        dropout=0.1,
        last_activation=False # last activation to be used on the output layer
    ):
        # initilizaing parent class
        super(SimpleCNN, self).__init__()
        self.last_activation = last_activation
        self.input_dim = input_dim

        # First Conv block
        in_dim = input_dim # For Mnist images this will be 28*28
        k = kernel_size
        s = stride
        pad = (in_dim * (s - 1) + k - s)//2

        self.conv1 = nn.Conv2d(
            1, num_channels, kernel_size=k, stride=s, padding=pad
        )
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        out_dim = (in_dim - 2)//2 + 1

        # Second Conv block
        in_dim = out_dim
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
        out_dim = (in_dim - 2)//2 + 1

        # dropout layer
        self.dropout = nn.Dropout(p=dropout)
        in_dim = out_dim
        # Flat fully connected layer
        self.fc = nn.Linear(out_channels * out_dim * out_dim, num_classes)  # MNIST images are 28x28

        if self.last_activation:
            self.last_activation = get_activation_function(self.last_activation)

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
        
        if self.last_activation:
            x = self.last_activation(x)
        return x