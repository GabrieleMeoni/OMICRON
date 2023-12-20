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

# add GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0" # GPU index

# import functions from train-test.py
from traintest import train
from traintest import test




# add data & utils folder to path variable (environment)
sys.path.insert(1, os.path.join("..", "data"))
sys.path.insert(1, os.path.join("..", "utils"))

# from files
from data_utils import Dataset
from plot_utils import plot_image

def main():
    # Path to the data folder (update the variable to your path).
    path_data=os.path.join("..", "data")
    # Seed value
    seed=22

    dataset=Dataset(path_data=path_data, seed=seed)
    dataset.read_data()

    dataset.get_statistics()

    batch_size=16

    # Train loader
    train_loader = DataLoader(dataset.get_split("train"), batch_size=batch_size, pin_memory=False, shuffle=True)
    # Cross validation data loader
    valid_loader = DataLoader(dataset.get_split("valid"), batch_size=batch_size, pin_memory=False, shuffle=True)
    # Test data loader
    test_loader = DataLoader(dataset.get_split("test"), batch_size=batch_size, pin_memory=False, shuffle=True)



        
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
        
    import torch.optim as optim
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0003)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    train(model, optimizer, torch.nn.CrossEntropyLoss(),train_loader, valid_loader, epochs=500, device=device)

    test(model, test_loader, torch.nn.CrossEntropyLoss(), device=device)
    return 

if __name__ == "__main__": 
    main()

