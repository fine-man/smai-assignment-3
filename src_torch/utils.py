import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .classifiers.mnist_cnn import SimpleCNN
import copy
import wandb

def evaluate(model, eval_dataset, batch_size=100, return_loss=True, criterion=nn.CrossEntropyLoss(), return_accuracy=True, device='cpu'):
    num_samples = len(eval_dataset) # number of examples
    
    model.to(device)
    # creating a Dataloader
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    it = 0 # iteration number
    accuracy = 0
    num_correct_preds = 0
    total_loss = 0

    # Find the predictions on the batches
    with torch.no_grad():
        model.eval()
        for X_minibatch, y_minibatch in eval_loader:
            # putting the data on device
            X_minibatch = X_minibatch.to(device)
            y_minibatch = y_minibatch.to(device)

            logits = model(X_minibatch)

            # model predictions
            y_pred = torch.argmax(logits, axis=1)

            if return_accuracy:
                num_correct_preds += torch.sum(y_pred == y_minibatch).cpu().item()

            if return_loss:
                # model loss
                loss = criterion(logits, y_minibatch).cpu().item()
                total_loss += loss * X_minibatch.shape[0]

    if return_accuracy and return_loss:
        # calculate accuracy
        accuracy = num_correct_preds/num_samples
        loss = total_loss/num_samples
        return accuracy, loss
    elif return_accuracy:
        accuracy = num_correct_preds/num_samples
        return accuracy, None
    elif return_loss:
        loss = total_loss/num_samples
        return None, loss
    else:
        return None, None

def predict(model, eval_dataset, batch_size=100, device='cpu', return_true_labels=True):
    num_samples = len(eval_dataset) # number of examples
    
    model.to(device)
    print(f"Model is on device: {device}")

    # creating a Dataloader
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    num_iterations = len(eval_loader)
    it = 0 # iteration number
    true_preds = []
    preds = []

    print(f"Total number of iterations: {num_iterations}")

    model.eval()
    with torch.no_grad():
        for i, (X_minibatch, y_minibatch) in enumerate(eval_loader):
            X_minibatch = X_minibatch.to(device)
            y_minibatch = y_minibatch.to(device)

            logits = model(X_minibatch)

            # model predictions
            y_pred = torch.argmax(logits, axis=1).detach().cpu()
            preds.append(y_pred)

            if return_true_labels:
                true_preds.append(y_minibatch.detach().cpu())
    
    preds = torch.concatenate(preds, dim=0)
    if return_true_labels:
        true_preds = torch.concatenate(true_preds, dim=0)
        return preds, true_preds

    return preds


def train(model, criterion, optimizer, train_dataset, val_dataset, **kwargs):
    # unpack keyword arguments
    batch_size = kwargs.pop("batch_size", 100)
    num_epochs = kwargs.pop("epochs", 10)
    print_every = kwargs.pop("print_every", 10)
    verbose = kwargs.pop("verbose", True)
    log_wandb = kwargs.pop("log_wandb", False)
    calc_accuracy = kwargs.pop("calc_accuracy", True)
    device = kwargs.pop("device", "cpu")

    model.to(device)
    print(f"\nModel is on device: {device}\n", flush=True)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    best_val_acc = 0
    best_val_loss = float('inf')
    best_params = None
    best_epoch = None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    iterations_per_epoch = len(train_loader)
    num_iterations = iterations_per_epoch * num_epochs
    num_train_samples = len(train_dataset)

    print(f"Number of Iterations Per Epoch: {iterations_per_epoch}", flush=True)

    it = 1 # current iteration number

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        # creating the minibatch loader
        model.train()
        for X_minibatch, y_minibatch in train_loader:
            X_minibatch = X_minibatch.to(device)
            y_minibatch = y_minibatch.to(device)
            # forward pass
            logits = model(X_minibatch)

            # calculate loss
            loss = criterion(logits, y_minibatch)

            # backward pass
            loss.backward()

            # update the weights and biases by taking a step in gradient direction
            optimizer.step()
            optimizer.zero_grad()
            
            # print iteration number and loss
            if verbose and it % print_every == 0:
                print(f"Iteration: {it}/{num_iterations} | loss = {loss:.4f}", flush=True)
            it += 1
        
        # Calculating Training and Validation accuracy after every epoch
        model.eval()
        train_acc, train_loss = evaluate(
            model, train_dataset,
            batch_size=batch_size,
            return_loss=True,
            return_accuracy=calc_accuracy,
            criterion=criterion,
            device=device
        )
        val_acc, val_loss = evaluate(
            model, val_dataset,
            batch_size=batch_size,
            return_loss=True,
            return_accuracy=calc_accuracy,
            criterion=criterion,
            device=device
        )

        if calc_accuracy:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                loss_at_best_val = val_loss
                best_epoch = epoch
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

        # logging the train/val loss and accuracy
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        if calc_accuracy:
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

        if log_wandb:
            data_to_log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            }
            if calc_accuracy:
                data_to_log["train_acc"] = train_acc
                data_to_log["val_acc"] = val_acc

            wandb.log(data_to_log)

        if verbose is True:
            print(f"Epoch: {epoch} ", end='')
            if calc_accuracy:
                print(f"| Train Accuracy: {train_acc*100:.3f} | Val Accuracy: {val_acc*100:.3f}", end='')
            print(f"|  Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
            print(flush=True)
    
    if log_wandb:
        data_to_log = {
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss
        }

        wandb.log(data_to_log)

    # Printing Final performance
    print(flush=True)
    if calc_accuracy:
        print(f"BEST VAL ACCURACY : {best_val_acc*100:.4f} ", end='')
    print(f"Best Epoch: {best_epoch} | Val loss: {best_val_loss:.4f}", flush=True)

    # returning the train/val loss and accuracies
    if calc_accuracy:
        return train_acc_history, val_acc_history,\
            train_loss_history, val_loss_history
    else:
        return train_loss_history, val_loss_history

