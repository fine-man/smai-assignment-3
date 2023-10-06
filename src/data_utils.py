import numpy as np

def minibatch_generator(X, y, minibatch_size=100):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - minibatch_size + 1,
                            minibatch_size):
        minibatch_idx = indices[start_idx: start_idx + minibatch_size]
        yield X[minibatch_idx], y[minibatch_idx]