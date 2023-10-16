import numpy as np

def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None

    num_examples = x.shape[0]
    x_reshaped = x.reshape(num_examples, -1)
    out = np.dot(x_reshaped, w) + b

    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    # shape: (N, D)
    x_reshaped = x.reshape(x.shape[0], -1)

    # dout @ w.T = (N, M) * (M, D) = (N, D) = (N, d1, ..., d_k)
    dx = np.dot(dout, w.T).reshape(x.shape[0], *x.shape[1:])
    
    # x_reshaped.T @ dout = (D, N) * (N, M) = (D, M)
    dw = np.dot(x_reshaped.T, dout)

    # sum((N, M), axis=0) = (M, )
    db = np.sum(dout, axis=0)

    return dx, dw, db

def sigmoid_forward(x):
    """Computes the forward pass for a layer of Sigmoid.

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: out
    """
    out = None

    out = 1/(1 + np.exp(-x))
    cache = out
    return out, cache

def sigmoid_backward(dout, cache):
    """Computes the backpass pass for a layer of sigmoid

    Input:
        - dout: Upstream derivates, of any shape
        - cache: output of sigmoid, same shape as dout
    
    Returns:
        - dx: Gradient with respect to input x
    """
    out = cache
    dx = dout * out * (1 - out)
    return dx

def tanh_forward(x):
    """Computes the forward layer of tanh

    Input:
        - x: Input of any shape where x[i, j] denotes the score of the jth class
            for the ith example
    
    Returns:
        - out: output, of same shape as input
        - cache: out
    """
    out = None
    a = np.exp(x)
    b = np.exp(-x)
    out = (a - b)/(a + b)
    cache = out
    return out, cache

def tanh_backward(dout, cache):
    """Computes the backpass pass for a layer of tanh

    Input:
        - dout: Upstream derivates, of any shape
        - cache: output of sigmoid, same shape as dout
    
    Returns:
        - dx: Gradient with respect to input x
    """
    out = cache
    dx = dout * (1 - out * out)
    return dx

def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None

    out = np.maximum(0, x)

    cache = x
    return out, cache

def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache

    local_dx = np.where(x > 0, 1, 0)
    dx = local_dx * dout
    return dx

def get_activation_forward(activation='relu'):
    if activation == 'sigmoid':
        return sigmoid_forward
    elif activation == 'tanh':
        return tanh_forward
    elif activation == "relu":
        return relu_forward
    else:
        return None
    
def get_activation_backward(activation='relu'):
    if activation == 'sigmoid':
        return sigmoid_backward
    elif activation == 'tanh':
        return tanh_backward
    elif activation == "relu":
        return relu_backward
    else:
        return None

def softmax(x):
    """Computers the softmax probabilities

    Inputs:
        - x: logits of shape (N, C) where x[i, j] is the score for the jth
          class for the ith input.
        
    Returns:
        - prob: softmax probabilities of shape (N, C)
    """
    num_examples = x.shape[0]
    
    # exponential of the original scores
    # Shape : (N, C)
    scores_exp = np.exp(x)

    # probability of each class for each example
    # Shape : (N, C)
    prob = scores_exp/np.sum(scores_exp, axis=1, keepdims=True)

    return prob

def softmax_loss(x, y, return_grad=False):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    num_examples = x.shape[0]

    # probability of each class for each example
    # Shape : (N, C)
    prob = np.exp(x)/np.sum(np.exp(x), axis=1).reshape(-1, 1) 

    # Indicator function which is 1 when j == y[i] and 0 otherwise
    # Shape : (N, C)
    I_j_equals_yi = np.zeros(prob.shape)
    I_j_equals_yi[np.arange(num_examples), y] = 1

    loss = np.mean(-np.log(prob[np.arange(num_examples), y]))
    if return_grad:
        dx = (prob - I_j_equals_yi)/num_examples
        return loss, dx
    else:
        return loss

def MSELoss(x, y, return_grad=False):
    """Computes the Mean Squared Error loss and gradient.

    Inputs:
    - x: Input data, of shape (N,) where x[i] gives the predicted regresion value
        for the ith examples
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    loss = np.mean((y - x) ** 2)

    if return_grad:
        dx = (-2/N) * (y - x)
        return loss, dx
    
    return loss

def CrossEntropyLoss(x, y, return_grad=False):
    """Computes the loss and gradient for Cross Entropy loss.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N, C) where y[i, j] is the true class probability
        of jth class for the ith example. This can also be of shape (N, )

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N, C = x.shape[:2]

    if len(y.shape) == 1:
        y_ = np.zeros((N, C))
        y_[np.arange(N), y] = 1.0
        y = y_

    # shape (N, )
    loss = np.mean(-y * np.log(x))

    if return_grad:
        grad = np.zeros_like(x) # (N, C)
        grad[:, :] = -y/x
        return loss, grad
    return loss

def get_criterion(crit_name):
    if crit_name == "MSE":
        return MSELoss
    else:
        return softmax_loss