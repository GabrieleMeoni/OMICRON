import os 
import torch 
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image, ImageFile
import torch.optim as optim

# import functions from train_test_functionality.py
from train_test_functionality import train
from train_test_functionality import test


# add data & utils folder to path variable (environment)
sys.path.insert(1, os.path.join("..", "data"))
sys.path.insert(1, os.path.join("..", "utils"))

# from files
from data_utils import Dataset
from plot_utils import plot_image



# add GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0" # GPU index


# TODO maybe implement data dataprocessing


def main():
    
    # constants & Hyperparameters
    seed_value=22
    learnnig_rate = 0.0003
    mommentum_value = 0.9
    batch_size=16
    num_epochs = 500

    # path to the data folder (update the variable to your path).
    path_data=os.path.join("..", "data")

    # use gpu if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    ### dataloading

    dataset=Dataset(path_data=path_data, seed=seed_value)
    dataset.read_data()
    dataset.get_statistics()


    # Train loader
    train_loader = DataLoader(dataset.get_split("train"), batch_size=batch_size, pin_memory=False, shuffle=True)
   
    # Cross validation data loader
    valid_loader = DataLoader(dataset.get_split("valid"), batch_size=batch_size, pin_memory=False, shuffle=True)
    
    # Test data loader
    test_loader = DataLoader(dataset.get_split("test"), batch_size=batch_size, pin_memory=False, shuffle=True)


    ### loading model 
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
    optimizer = torch.optim.SGD(model.parameters(), lr=learnnig_rate, momentum=mommentum_value)

    # move model to GPU if available
    model.to(device)

    #### training
    train(model, optimizer, torch.nn.CrossEntropyLoss(),train_loader, valid_loader, epochs=num_epochs, device=device)


    #### testing
    # test(model, test_loader, torch.nn.CrossEntropyLoss(), device=device)




    return

if __name__ == "__main__": 
    main()

