import numpy as np
from src import *

class Solver():
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]

        # unpack keyword arguments
        self.update_rule = kwargs.pop("update_rule", "sgd")
        self.update_type = kwargs.pop("update_type", "minibatch")
        self.optim_config = kwargs.pop("optim_config", {})
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)
        self.num_train_samples = self.X_train.shape[0]
        self.num_val_samples = self.X_val.shape[0]

        # Throw an error if there are extra arguments
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError(f"Unrecognized arguments {extra}")
        
        # if update type is stochastic gradient descent, then make batchsize = 1
        if self.update_type == "stochastic":
            self.batch_size = 1

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if not hasattr(optim, self.update_rule):
            raise ValueError(f'Invalid update_rule "{self.update_rule}"')
        self.update_rule = getattr(optim, self.update_rule)

        # reset solver variables
        self._reset()
    
    def _reset(self):
        """
        Manually resetting some variables used for book-keeping
        """
        self.epoch = 0
        self.val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # make a deep copy of optimizer config for every parameter
        self.optim_configs = {}
        for param_name in self.model.params:
            default = {key:value for key, value in self.optim_config.items()}
            self.optim_configs[param_name] = default
    
    def _step(self):
        """
        Weight update step
        """
        grads = self.model.grads
        for param_name, param in self.model.params.items():
            w, dw = param, grads[param_name]
            config = self.optim_configs[param_name]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[param_name] = next_w
            self.optim_configs[param_name] = next_config

    def check_accuracy(self, X, y, batch_size=100, num_samples=None):
        N = X.shape[0]

        # Sub-sample the data
        if num_samples is not None and num_samples < N:
            mask = np.random.choice(N, num_samples)
            X = X[mask]
            y = y[mask]
            N = num_samples

        # Find the number of batches
        num_batches = max(N // batch_size, 1)
        if N % batch_size != 0:
            num_batches += 1
        it = 0
        acc = 0
        y_preds = []

        # Find the predictions on the batches
        self.model.eval()
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            logits = self.model.forward(X[start:end])
            y_pred = np.argmax(logits, axis=1)
            y_preds.append(y_pred)

        # calculate accuracy
        y_preds = np.concatenate(y_preds, axis=0)
        acc = np.mean(y_preds == y)
        return acc

    def train(self):
        batch_size, num_epochs = self.batch_size, self.num_epochs

        iterations_per_epoch = max(self.num_train_samples // batch_size, 1)
        if self.num_train_samples % batch_size != 0:
            iterations_per_epoch += 1
        num_iterations = iterations_per_epoch * num_epochs
        it = 1

        for epoch in range(1, num_epochs + 1):
            self.model.zero_grad()
            minibatch_gen = minibatch_generator(self.X_train, self.y_train, batch_size)
            self.model.train()
            for X_minibatch, y_minibatch in minibatch_gen:
                # forward pass
                logits = self.model.forward(X_minibatch)

                # calculate loss
                loss, dout = softmax_loss(logits, y_minibatch, return_grad=True)
                self.loss_history.append(loss)

                # dividing the softmax gradient by total number of samples
                # when performing batch gradient descent
                if self.update_type == "batch":
                    dout *= self.batch_size/self.num_train_samples

                # backward pass
                self.model.backward(dout)

                # update the weights and biases by taking a step in gradient direction
                if self.update_type in ["stochastic", "minibatch"]:
                    self._step()
                    self.model.zero_grad()
                
                # print iteration number and loss
                if self.verbose and it % self.print_every == 0:
                    print(f"Iteration: {it}/{num_iterations} | loss = {loss}")
                it += 1
            
            if self.update_type == "batch":
                self._step()
                self.model.zero_grad()
            
            self.epoch += 1

            # Calculating Training and Validation accuracy after every epoch
            self.model.eval()
            train_acc = self.check_accuracy(
                self.X_train, self.y_train,
                batch_size=self.batch_size
            )
            val_acc = self.check_accuracy(
                self.X_val, self.y_val,
                batch_size=self.batch_size
            )

            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)

            if self.verbose is True:
                print(f"Epoch: {self.epoch} | Train Accuracy: {train_acc*100:.3f} | Validation Accuracy: {val_acc*100:.3f}")