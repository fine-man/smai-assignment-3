import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .classifiers import SimpleCNN, SimpleMLP
import copy
import wandb

def get_model_type(config):
    if "conv_layers" in config.keys():
        return "CNN"
    else:
        return "MLP"

def get_model_mlp(config):
    input_dim = config.pop("input_dim")
    num_classes = config.pop("num_classes")

    # Number of layers
    num_layers = config.pop("num_layers", 1)
    hidden_dims = []

    for i in range(1, num_layers + 1):
        dim = config.pop(f"hidden_dims{i}")
        hidden_dims.append(dim)
    
    model = SimpleMLP(input_dim, hidden_dims, num_classes, **config)
    return model

def get_model_cnn(config):
    input_dim = config.pop("input_dim", 28)
    num_classes = config.pop("num_classes", 10)
    num_conv_layers = config.pop("conv_layers", 2)
    
    model = SimpleCNN(input_dim, num_classes, **config)
    return model

def get_criterion(crit_name):
    if crit_name == "CE":
        return nn.CrossEntropyLoss()
    elif crit_name == "BCE":
        return nn.BCELoss()
    else:
        return nn.CrossEntropyLoss()

def get_optimizer(model, config):
    lr = config["learning_rate"]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer

def trigger_training(config, train_dataset, val_dataset):
    torch.manual_seed(42)

    # getting the model, criterion and optimizer
    model_type = get_model_type(config["model"])
    print(f"Model Type: {model_type}")
    if model_type == "CNN":
        model = get_model_cnn(config["model"])
    else:
        model = get_model_mlp(config["model"])

    print(model, flush=True)
    criterion = get_criterion(config["criterion"])
    optimizer = get_optimizer(model, config["optimizer"])

    # training config
    train_config = config["training"]
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    train(
        model, criterion, optimizer, train_dataset, val_dataset, device=device, **train_config)
    
    return model

def accuracy_score(y_true, y_pred, normalize=True):
    if len(y_true.shape) > 1:
        eval_type = 'multilabel'
    else:
        eval_type = 'multiclass'

    if eval_type == 'multilabel':
        y_pred[y_pred < 0.5] = 0
        y_pred[y_pred >= 0.5] = 1
        
        # considering correctness only when all labels are correct
        num_correct = torch.all((y_pred == y_true), dim=1).sum().cpu().item() 
        
        if normalize:
            accuracy = num_correct/y_pred.shape[0]
            return accuracy
        else:
            return num_correct
    else:
        num_correct = torch.sum(y_pred == y_true).cpu().item()
        if normalize:
            accuracy = num_correct/y_pred.shape[0]
            return accuracy
        else:
            return num_correct

def evaluate(model, eval_dataset, **kwargs):
    batch_size = kwargs.pop("batch_size", 100)
    argmax = kwargs.pop("argmax", True)
    return_loss = kwargs.pop("return_loss", True)
    criterion = kwargs.pop("criterion", nn.CrossEntropyLoss())
    return_accuracy = kwargs.pop("return_accuracy", True)
    accuracy_func = kwargs.pop("accuracy_func", accuracy_score)
    device = kwargs.pop("device", "cpu")

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
            if argmax:
                y_pred = torch.argmax(logits, axis=1)
            else:
                y_pred = logits

            if return_accuracy:
                num_correct_preds += accuracy_func(y_minibatch, y_pred, normalize=False)

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

def predict(model, eval_dataset, batch_size=100, device='cpu', argmax=True, return_true_labels=True):
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
            if argmax:
                y_pred = torch.argmax(logits, axis=1).detach().cpu()
            else:
                y_pred = logits.detach().cpu()

            preds.append(y_pred)

            if return_true_labels:
                true_preds.append(y_minibatch.detach().cpu())
    
    preds = torch.concatenate(preds, dim=0)
    if return_true_labels:
        true_preds = torch.concatenate(true_preds, dim=0)
        return preds, true_preds

    return preds

def train(model, criterion, optimizer, train_dataset, val_dataset=None, **kwargs):
    # unpack keyword arguments
    batch_size = kwargs.pop("batch_size", 100)
    num_epochs = kwargs.pop("epochs", 10)
    print_every = kwargs.pop("print_every", 10)
    verbose = kwargs.pop("verbose", True)
    log_wandb = kwargs.pop("log_wandb", False)
    calc_accuracy = kwargs.pop("calc_accuracy", True)
    device = kwargs.pop("device", "cpu")
    argmax = kwargs.pop("argmax", True)
    accuracy_func = kwargs.pop("accuracy_func", accuracy_score)

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
            if (verbose and it % print_every == 0) or (it == 1):
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
            device=device,
            argmax=argmax,
            accuracy_func=accuracy_func,
        )

        if val_dataset:
            val_acc, val_loss = evaluate(
                model, val_dataset,
                batch_size=batch_size,
                return_loss=True,
                return_accuracy=calc_accuracy,
                criterion=criterion,
                device=device,
                argmax=argmax,
                accuracy_func=accuracy_func
            )

        if calc_accuracy:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
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
        print(f"BEST VAL ACCURACY : {best_val_acc*100:.4f} | ", end='')
    print(f"Best Epoch: {best_epoch} | Val loss: {best_val_loss:.4f}", flush=True)

    # returning the train/val loss and accuracies
    if calc_accuracy:
        return train_acc_history, val_acc_history,\
            train_loss_history, val_loss_history
    else:
        return train_loss_history, val_loss_history
