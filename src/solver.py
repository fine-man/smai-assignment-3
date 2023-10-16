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
        self.train_loss_history = []
        self.val_loss_history = []
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

    def check_accuracy(self, X, y, batch_size=100, num_samples=None, return_loss=False):
        N = X.shape[0] # number of examples

        # Sub-sample the data
        if num_samples is not None and num_samples < N:
            mask = np.random.choice(N, num_samples)
            X = X[mask]
            y = y[mask]
            N = num_samples

        # creating a minibatch generator
        minibatch_gen = minibatch_generator(X, y, batch_size)
        it = 0 # iteration number
        accuracy = 0
        y_preds = []
        total_loss = 0

       # Find the predictions on the batches
        self.model.eval()
        for X_minibatch, y_minibatch in minibatch_gen:
            logits = self.model.forward(X_minibatch)

            # model predictions
            y_pred = np.argmax(logits, axis=1)
            y_preds.append(y_pred)

            if return_loss:
                # model loss
                loss = softmax_loss(logits, y_minibatch)
                total_loss += loss * X_minibatch.shape[0]

        # calculate accuracy
        y_preds = np.concatenate(y_preds, axis=0)
        accuracy = np.mean(y_preds == y)

        if return_loss:
            loss = total_loss/X.shape[0]
            return accuracy, loss

        return accuracy

    def train(self):
        batch_size, num_epochs = self.batch_size, self.num_epochs

        iterations_per_epoch = max(self.num_train_samples // batch_size, 1)
        if self.num_train_samples % batch_size != 0:
            iterations_per_epoch += 1
        # total number of iterations
        num_iterations = iterations_per_epoch * num_epochs
        it = 1 # current iteration number

        for epoch in range(1, num_epochs + 1):
            self.model.zero_grad()

            minibatch_gen = minibatch_generator(self.X_train, self.y_train, batch_size)
            self.model.train()
            for X_minibatch, y_minibatch in minibatch_gen:
                # forward pass
                logits = self.model.forward(X_minibatch)

                # calculate loss
                loss, dout = softmax_loss(logits, y_minibatch, return_grad=True)

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
                    print(f"Iteration: {it}/{num_iterations} | loss = {loss:.4f}")
                it += 1
            
            if self.update_type == "batch":
                self._step()
                self.model.zero_grad()
            
            self.epoch += 1

            # Calculating Training and Validation accuracy after every epoch
            self.model.eval()
            train_acc, train_loss = self.check_accuracy(
                self.X_train, self.y_train,
                batch_size=self.batch_size,
                return_loss=True
            )
            val_acc, val_loss = self.check_accuracy(
                self.X_val, self.y_val,
                batch_size=self.batch_size,
                return_loss=True
            )

            # logging the train/val loss and accuracy
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)

            if self.verbose is True:
                print(f"Epoch: {self.epoch} | Train Accuracy: {train_acc*100:.3f} | Val Accuracy: {val_acc*100:.3f} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
                print()