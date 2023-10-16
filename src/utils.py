import numpy as np
import copy
from src import *
from src.classifiers import *
from .layers import get_criterion
import wandb

def check_accuracy(model, X, y, batch_size=100, num_samples=None, return_loss=False, criterion=softmax_loss):
    N = X.shape[0] # number of examples

    # Sub-sample the data
    if num_samples is not None and num_samples < N:
        mask = np.random.choice(N, num_samples)
        X = X[mask]
        y = y[mask]
        N = num_samples
    
    num_samples = N
    # creating a minibatch generator
    minibatch_gen = minibatch_generator(X, y, batch_size)
    it = 0 # iteration number
    accuracy = 0
    num_correct_preds = 0
    total_loss = 0

    # Find the predictions on the batches
    model.eval()
    for X_minibatch, y_minibatch in minibatch_gen:
        logits = model.forward(X_minibatch)

        # model predictions
        y_pred = np.argmax(logits, axis=1)
        num_correct_preds += np.sum(y_pred == y_minibatch)

        if return_loss:
            # model loss
            loss = criterion(logits, y_minibatch)
            total_loss += loss * X_minibatch.shape[0]

    # calculate accuracy
    accuracy = num_correct_preds/num_samples

    if return_loss:
        loss = total_loss/X.shape[0]
        return accuracy, loss

    return accuracy

def train(model, criterion, optimizer, X_train, y_train, X_val, y_val, **kwargs):
    # unpack keyword arguments
    update_type = kwargs.pop("update_type", "minibatch")
    batch_size = kwargs.pop("batch_size", 100)
    num_epochs = kwargs.pop("epochs", 10)
    print_every = kwargs.pop("print_every", 10)
    verbose = kwargs.pop("verbose", True)
    log_wandb = kwargs.pop("log_wandb", False)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    best_val_acc = 0
    loss_at_best_val = None
    best_params = None
    best_epoch = None
    
    num_train_samples = X_train.shape[0]
    iterations_per_epoch = max(num_train_samples // batch_size, 1)
    if num_train_samples % batch_size != 0:
        iterations_per_epoch += 1
    # total number of iterations
    num_iterations = iterations_per_epoch * num_epochs
    it = 1 # current iteration number

    for epoch in range(1, num_epochs + 1):
        model.zero_grad()

        # creating the minibatch loader
        minibatch_gen = minibatch_generator(X_train, y_train, batch_size)
        model.train()
        for X_minibatch, y_minibatch in minibatch_gen:
            # forward pass
            logits = model.forward(X_minibatch)

            # calculate loss
            loss, dout = criterion(logits, y_minibatch, return_grad=True)

            # dividing the softmax gradient by total number of samples
            # when performing batch gradient descent
            if update_type == "batch":
                dout *= batch_size/num_train_samples

            # print("Just before a backward pass")
            # backward pass
            model.backward(dout)

            # update the weights and biases by taking a step in gradient direction
            if update_type in ["stochastic", "minibatch"]:
                optimizer.step()
                model.zero_grad()
            
            # print iteration number and loss
            if verbose and it % print_every == 0:
                print(f"Iteration: {it}/{num_iterations} | loss = {loss:.4f}")
            it += 1
        
        if update_type == "batch":
            optimizer.step()
            model.zero_grad()
        
        # Calculating Training and Validation accuracy after every epoch
        model.eval()
        train_acc, train_loss = check_accuracy(
            model, X_train, y_train,
            batch_size=batch_size,
            return_loss=True,
            criterion=criterion
        )
        val_acc, val_loss = check_accuracy(
            model, X_val, y_val,
            batch_size=batch_size,
            return_loss=True,
            criterion=criterion
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            loss_at_best_val = val_loss
            best_params = copy.deepcopy(model.parameters())
            best_epoch = epoch

        # logging the train/val loss and accuracy
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        if log_wandb:
            data_to_log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc
            }
            wandb.log(data_to_log)

        if verbose is True:
            print(f"Epoch: {epoch} | Train Accuracy: {train_acc*100:.3f} | Val Accuracy: {val_acc*100:.3f} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
            print()
    
    if log_wandb:
        data_to_log = {
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "loss_at_best_val": loss_at_best_val
        }

    print(f"\nBEST VAL ACCURACY : {best_val_acc*100:.4f} | epoch: {best_epoch} | Val loss: {loss_at_best_val:.4f}")
    model.load_params(best_params)
    print(f"Best Parameters have been loaded in the model")

    # returning the train/val loss and accuracies
    return train_acc_history, val_acc_history,\
        train_loss_history, val_loss_history

def trigger_training(config, X_train, y_train, X_val, y_val):
    np.random.seed(42)
    # getting the model, criterion and optimizer
    model = get_model(config["model"])
    print(model.parameters().keys())
    criterion = get_criterion(config["criterion"])
    optimizer = get_optimizer(config["optimizer"], model)

    # training config
    train_config = config["training"]
    
    train(
        model, criterion, optimizer, X_train, y_train, X_val, y_val, **train_config)
    
    return model

def get_model(config):
    input_dim = config.pop("input_dim")
    num_classes = config.pop("num_classes")
    activation = config.pop("activation")

    # Number of layers
    num_layers = config.pop("num_layers", 1)
    hidden_dims = []
    hidden_dims.append(input_dim)

    for i in range(1, num_layers + 1):
        dim = config.pop(f"hidden_dims{i}")
        hidden_dims.append(dim)
    
    print(hidden_dims)

    model = FullyConnectedNet(
        input_dim, hidden_dims, num_classes, 
        activation, **config)
    return model

def get_optimizer(config, model):
    lr = config["learning_rate"]
    update_rule = config["update_rule"]
    optimizer = Optimizer(model, update_rule, optim_config={"learning_rate": lr})
    return optimizer
