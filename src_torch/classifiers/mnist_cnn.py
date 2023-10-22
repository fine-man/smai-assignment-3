import torch
import torch.nn as nn

# Define a Simple CNN architecture for MNIST
class SimpleCNN(nn.Module):
    def __init__(
        self, 
        num_classes=10,
        dropout=0.25):
        # initilizaing parent class
        super(SimpleCNN, self).__init__()
        # First Conv block
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second Conv block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # dropout layer
        self.dropout = nn.Dropout(p=dropout)
        # Flat fully connected layer
        self.fc = nn.Linear(32 * 7 * 7, num_classes)  # MNIST images are 28x28

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