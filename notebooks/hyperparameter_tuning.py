import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.trial import TrialState
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import v2
from torchvision import datasets, transforms

import numpy as np

import sys
sys.path.insert(1, os.path.join("..", "data"))
sys.path.insert(1, os.path.join("..", "utils"))

from PIL import Image, ImageFile
from datetime import datetime
import matplotlib.pyplot as plt
import torchvision.models as models

from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline

def read_data():
    # Path to the data folder (update the variable to your path).
    path_data=os.path.join("..", "data")
    # Seed value
    seed=22
    torch.manual_seed(seed)

    image_width = 1024
    image_height = 1942
    scale_factor = 0.1

    valid_size = 0.15
    test_size = 0.15

    dataset_mean = [0.3895,0.3895,0.3895]
    dataset_std = [0.1563,0.1563,0.1563]

    transform = transforms.Compose([v2.ToImage(),
                                    v2.Resize((int(256), int(256)), antialias=True),
                                    v2.RandomHorizontalFlip(p=0.5),
                                    v2.RandomVerticalFlip(p=0.5),
                                    v2.ToDtype(torch.float32, scale=True),
                                    v2.Grayscale(num_output_channels=1),
                                    v2.Normalize((dataset_mean),(dataset_std))
                                    ])

    dataset = datasets.ImageFolder(root=path_data, 
                                     transform=transform)

    n_val = int(np.floor(valid_size * len(dataset)))
    n_test = int(np.floor(test_size * len(dataset)))
    n_train = len(dataset) - n_val - n_test

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
    
    return train_ds,val_ds,test_ds

def train(model, optimizer, loss_fn, train_loader, val_loader):
    training_loss = 0.0
    valid_loss = 0.0
    model.train()
    for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)

    training_loss /= len(train_loader.dataset)

    model.eval()
    num_correct = 0
    num_examples = 0
    correct = 0 
    accuracy = 0
    with torch.no_grad():

        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        accuracy = num_correct / num_examples
        
    return accuracy

def test(network, test_loader):
    """Tests the model.

    Parameters:
        - network (__main__.Net): The CNN

    Returns:
        - accuracy_test (torch.Tensor): The test accuracy
    """
    network.eval()         # Set the module in evaluation mode (only affects certain modules)
    correct = 0
    with torch.no_grad():  # Disable gradient calculation (when you are sure that you will not call Tensor.backward())
        for batch_i, (data, target) in enumerate(test_loader):  # For each batch

            # Limit testing data for faster computation
            if batch_i * 32 > number_of_test_examples:
                break

            output = network(data.to(device))               # Forward propagation
            pred = output.data.max(1, keepdim=True)[1]      # Find max value in each row, return indexes of max values
            correct += pred.eq(target.to(device).data.view_as(pred)).sum()  # Compute correct predictions

    accuracy_test = correct / len(test_loader.dataset)

    return accuracy_test

def objective(trial):
    """Objective function to be optimized by Optuna.

    Hyperparameters chosen to be optimized: optimizer, learning rate,
    dropout values, number of convolutional layers, number of filters of
    convolutional layers, number of neurons of fully connected layers.

    Inputs:
        - trial (optuna.trial._trial.Trial): Optuna trial
    Returns:
        - accuracy(torch.Tensor): The test accuracy. Parameter to be maximized.
    """
    # Generate the model
    model = models.mobilenet_v3_small(pretrained=True)

    # Generate the optimizer
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True) 
    mom = trial.suggest_float('mom', 0,1,step =0.1)
    wd = trial.suggest_float("wd", 1e-4, 1e-1, log=True)
    dmp = trial.suggest_float('dmp', 0,1,step=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, dampening=dmp, weight_decay=wd)
    
    batch_size = trial.suggest_categorical('batch_size',[2,4,8,16,32])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers = 4)
    # Cross validation data loader
    valid_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers = 4)
    # Test data loader
    test_loader = DataLoader(test_ds, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers = 4)

    # Training of the model
    for epoch in range(n_epochs):
        accuracy =train(model, optimizer, torch.nn.CrossEntropyLoss(), train_loader, valid_loader)  # Train the model
        print(accuracy)

        # For pruning (stops trial early if not promising)
        trial.report(accuracy, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            
            raise optuna.exceptions.TrialPruned()

    return accuracy

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Optimization study for a PyTorch CNN with Optuna
    # -------------------------------------------------------------------------

    # Use cuda if available for faster computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Parameters ----------------------------------------------------------
    n_epochs = 30                         # Number of training epochs
    number_of_trials = 1000                # Number of Optuna trials
    limit_obs = True                      # Limit number of observations for faster computation

    # *** Note: For more accurate results, do not limit the observations.
    #           If not limited, however, it might take a very long time to run.
    #           Another option is to limit the number of epochs. ***

    if limit_obs:  # Limit number of observations
        number_of_train_examples = 500 * 32  # Max train observations
        number_of_test_examples = 5 * 1000      # Max test observations
    else:
        number_of_train_examples = 60000                   # Max train observations
        number_of_test_examples = 10000                    # Max test observations
    # -------------------------------------------------------------------------

    # Make runs repeatable
    random_seed = 22
    torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
    torch.manual_seed(random_seed)


    # Download MNIST dataset to 'files' directory and normalize it
    train_ds,val_ds,test_ds = read_data()

    # Create an Optuna study to maximize test accuracy
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=number_of_trials)

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    # Find number of pruned and completed trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Display the study statistics
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save results to csv file
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
    df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on accuracy
    df.to_csv('optuna_results.csv', index=False)  # Save to csv file

    # Display results in a dataframe
    print("\nOverall Results (ordered by accuracy):\n {}".format(df))

    # Find the most important hyperparameters
    most_important_parameters = optuna.importance.get_param_importances(study, target=None)

    # Display the most important hyperparameters
    print('\nMost important hyperparameters:')
    for key, value in most_important_parameters.items():
        print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))
		
    fig1=plot_optimization_history(study)
    fig1.save("optimization.png")
    
    fig2=plot_intermediate_values(study)
    fig2.save("intermediate.png")

    fig3=plot_contour(study)
    fig3.save("contour.png")

    fig4=plot_param_importances(study)
    fig4.save("importances.png")

    fig5=plot_rank(study)
    fig5.save("rank.png")

    fig6=plot_timeline(study)
    fig6.save("timeline.png")