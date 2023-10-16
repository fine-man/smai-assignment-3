import numpy as np
from ..layers import *
import copy

class FullyConnectedNet():
    """Class for multi-layer fully connected Neural Network

    Architecture:
        - {affine - activation_layer} x (L - 1) - affine 
    """
    def __init__(
        self,
        input_dim,
        hidden_dims,
        num_classes,
        activation='relu',
        last_activation=None, # activation to be used after last layer
        dtype=np.float32,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims) + 1
        self.activation = activation
        self.last_activation = last_activation
        self.dtype = dtype
        
        self.mode = 'train'
        self.params = {}
        self.cache = {}
        self.grads = {}

        # getting the activation function
        self.activation_forward = get_activation_forward(activation)
        self.activation_backward = get_activation_backward(activation)

        # Activation to be applied after the last affine layer
        if self.last_activation:
            self.last_activation_forward = get_activation_forward(last_activation)
            self.last_activation_backward = get_activation_backward(last_activation)

        # first layer
        if self.num_layers > 1:
            self.params["W1"] = np.random.randn(input_dim, hidden_dims[0])
            self.grads["W1"] = None
            self.params["b1"] = np.random.randn(hidden_dims[0])
            self.grads["b1"] = None

        # intermediate layers
        for i in range(1, len(hidden_dims)):
            j = i + 1 # index of the layer
            in_dim = hidden_dims[i - 1]
            out_dim = hidden_dims[i]
            
            self.params[f"W{j}"] = np.random.randn(in_dim, out_dim)
            self.grads[f"W{j}"] = None
            self.params[f"b{j}"] = np.random.randn(out_dim)
            self.grads[f"b{j}"] = None

        # last layer
        in_dim = hidden_dims[-1] if self.num_layers > 1 else input_dim
        out_dim = num_classes
        j = len(hidden_dims) + 1
        self.params[f"W{j}"] = np.random.randn(in_dim, out_dim)
        self.grads[f"W{j}"] = None
        self.params[f"b{j}"] = np.random.randn(out_dim)
        self.grads[f"b{j}"] = None

        for name, param in self.params.items():
            self.params[name] = param.astype(self.dtype)
    
    def eval(self, value=True):
        if value is True:
            self.mode = 'test'
        else:
            self.mode = 'train'
    
    def train(self, value=True):
        if value is True:
            self.mode = 'train'
        else:
            self.mode = 'test'
    
    def parameters(self):
        return self.params
    
    def load_params(self, params):
        self.params = copy.deepcopy(params)
    
    def zero_grad(self, set_to_none=True):
        for name, grad in self.grads.items():
            if set_to_none:
                self.grads[name] = None
            else:
                self.grads[name] = np.zeros_like(self.params[name])

    def forward(self, x):
        cache = {}
        activ = self.activation
        last_activ = self.last_activation
        activation_forward = self.activation_forward
        if self.last_activation:
            last_activation_forward = self.last_activation_forward
        out = x

        # forwarding through all layers except the last one
        for i in range(self.num_layers - 1):
            j = i + 1
            w, b = self.params[f"W{j}"], self.params[f"b{j}"]
            if self.mode == 'train':
                out, cache[f"affine{j}"] = affine_forward(out, w, b)
                out, cache[f"{activ}{j}"] = activation_forward(out)
            else:
                out, _ = affine_forward(out, w, b)
                out, _ = activation_forward(out)
        
        # last layer
        j = self.num_layers
        w, b = self.params[f"W{j}"], self.params[f"b{j}"]

        if self.mode == "train":
            out, cache[f"affine{j}"] = affine_forward(out, w, b)
            if self.last_activation:
                out, cache[f"{last_activ}"] = last_activation_forward(out)
            self.cache = cache
        else:
            out, _ = affine_forward(out, w, b)
            if self.last_activation:
                out, _ = last_activation_forward(out)

        return out
    
    def backward(self, dout):
        if self.mode == 'test':
            return

        grads = self.grads
        cache = self.cache
        j = self.num_layers
        activ = self.activation
        last_activ = self.last_activation
        activation_backward = self.activation_backward
        if self.last_activation:
            last_activation_backward = self.last_activation_backward

        # backward pass of the last activation layer if it exists
        if self.last_activation:
            dout = last_activation_backward(dout, cache[f"{last_activ}"])

        # backward pass of the last affine layer
        dout, dw, db = affine_backward(dout, cache[f"affine{j}"])
        grads[f"W{j}"] = np.copy(dw) if grads[f"W{j}"] is None else grads[f"W{j}"] + dw
        grads[f"b{j}"] = np.copy(db) if grads[f"b{j}"] is None else grads[f"b{j}"] + db

        # backward pass through all the other layers
        for j in range(self.num_layers - 1, 0, -1):
            # backprop through activation layer
            dout = activation_backward(dout, cache[f"{activ}{j}"])
            dout, dw, db = affine_backward(dout, cache[f"affine{j}"])
            grads[f"W{j}"] = np.copy(dw) if grads[f"W{j}"] is None else grads[f"W{j}"] + dw
            grads[f"b{j}"] = np.copy(db) if grads[f"b{j}"] is None else grads[f"b{j}"] + db
        
        self.grads = grads