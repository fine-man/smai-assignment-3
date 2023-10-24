import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dims,
            num_classes,
            flatten_first=False,
            last_activation=False, # activation to be used after last layer
    ):
        super(SimpleMLP, self).__init__()
        self.input_dim = input_dim
        self.num_layers = len(hidden_dims) + 1
        self.hidden_dims = hidden_dims
        self.flatten_first = flatten_first
        self.last_activation = last_activation 
        modules = []

        # first layer
        in_dim = input_dim
        if self.num_layers > 1:
            output_dim = hidden_dims[0]
            modules.append(nn.Linear(in_dim, output_dim))
            modules.append(nn.ReLU())
            in_dim = output_dim
        
        # Intermediate layers
        for i in range(1, self.num_layers - 1):
            out_dim = hidden_dims[i]
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(nn.ReLU())
            in_dim = out_dim
        
        # Last Layer
        out_dim = num_classes
        modules.append(nn.Linear(in_dim, out_dim))
        if self.last_activation:
            modules.append(nn.ReLU())
        
        self.modules = nn.ModuleList(modules)
        
    def forward(self, x):
        N = x.shape[0] # Number of samples
        if self.flatten_first:
            out = x.reshape(N, -1)
        else:
            out = x

        for module in self.modules:
            out = module(out)
        
        return out