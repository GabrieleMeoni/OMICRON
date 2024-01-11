import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
import os # modified

batch_size=16

image_width = 1024
image_height = 1942
scale_factor = 0.4

valid_size = 0.15
test_size = 0.15

dataset_mean = [0.2391, 0.4028, 0.4096]
dataset_std = [0.2312, 0.3223, 0.3203]

path_data=os.path.join("..", "data") # modified


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((dataset_mean),(dataset_std)),
                                transforms.Resize((int(image_width*scale_factor), int(image_height*scale_factor)))])

dataset = datasets.ImageFolder(root=path_data, 
                                 transform=transform)

n_val = int(np.floor(valid_size * len(dataset)))
n_test = int(np.floor(test_size * len(dataset)))
n_train = len(dataset) - n_val - n_test

train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

# Train loader
train_loader = DataLoader(train_ds, batch_size=batch_size, pin_memory=False, shuffle=True)
# train_features, train_labels = next(iter(train_loader)) # modified
# print(train_labels) # modified

# Cross validation data loader
valid_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=False, shuffle=True)
# Test data loader
test_loader = DataLoader(test_ds, batch_size=batch_size, pin_memory=False, shuffle=True)

#0 - cloud
#1 - edge
#2 - good

unique, counts = np.unique(torch.tensor([train_ds.dataset.targets[i] for i in train_ds.indices]), return_counts=True)
print("Train split: ", dict(zip(unique, counts)))

unique, counts = np.unique(torch.tensor([test_ds.dataset.targets[i] for i in test_ds.indices]), return_counts=True)
print("Test split: ", dict(zip(unique, counts)))

unique, counts = np.unique(torch.tensor([val_ds.dataset.targets[i] for i in val_ds.indices]), return_counts=True)
print("Validation split: ", dict(zip(unique, counts)))